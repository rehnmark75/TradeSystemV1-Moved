-- Virtual Stop Loss Columns for Scalping Mode
-- This migration adds columns to support streaming-based virtual stop loss monitoring
-- for scalp trades where IG's minimum SL restrictions are too wide.

-- Add columns to trade_log table
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS is_scalp_trade BOOLEAN DEFAULT FALSE;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS virtual_sl_pips FLOAT;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS virtual_sl_price FLOAT;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS virtual_sl_triggered BOOLEAN DEFAULT FALSE;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS virtual_sl_triggered_at TIMESTAMP;

-- Add index for efficient scalp trade queries
CREATE INDEX IF NOT EXISTS idx_trade_log_is_scalp_trade ON trade_log(is_scalp_trade) WHERE is_scalp_trade = TRUE;

-- Add index for VSL triggered trades
CREATE INDEX IF NOT EXISTS idx_trade_log_virtual_sl_triggered ON trade_log(virtual_sl_triggered) WHERE virtual_sl_triggered = TRUE;

-- Comment on columns for documentation
COMMENT ON COLUMN trade_log.is_scalp_trade IS 'True if this trade is in scalping mode with virtual stop loss monitoring';
COMMENT ON COLUMN trade_log.virtual_sl_pips IS 'Virtual stop loss distance in pips (e.g., 4.0 pips) - tighter than broker minimum';
COMMENT ON COLUMN trade_log.virtual_sl_price IS 'Calculated virtual stop loss price level based on entry + virtual_sl_pips';
COMMENT ON COLUMN trade_log.virtual_sl_triggered IS 'True if the virtual stop loss was triggered (position closed programmatically)';
COMMENT ON COLUMN trade_log.virtual_sl_triggered_at IS 'Timestamp when virtual stop loss was triggered and position closed';
