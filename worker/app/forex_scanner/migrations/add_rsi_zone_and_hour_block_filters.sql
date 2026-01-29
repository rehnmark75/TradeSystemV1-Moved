-- Migration: Add RSI Zone Block and Hour Block filters (per-pair)
-- Date: 2026-01-29
-- Purpose: Enable per-pair filtering of:
--   1. RSI zones that perform poorly (e.g., block BUY when RSI 50-60)
--   2. Specific hours that perform poorly (e.g., block hours 0,5,6,22,23 UTC)

-- Add RSI zone block columns
-- These define a "bad zone" where signals should be blocked
-- Example: scalp_rsi_block_buy_min=50, scalp_rsi_block_buy_max=60 blocks BUY when RSI is 50-60
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_rsi_block_buy_min DECIMAL(5,2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_rsi_block_buy_max DECIMAL(5,2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_rsi_block_sell_min DECIMAL(5,2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_rsi_block_sell_max DECIMAL(5,2) DEFAULT NULL;

-- Add blocked hours column (comma-separated UTC hours, e.g., "0,5,6,22,23")
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_blocked_hours_utc VARCHAR(100) DEFAULT NULL;

-- Add comments for documentation
COMMENT ON COLUMN smc_simple_pair_overrides.scalp_rsi_block_buy_min IS 'Minimum RSI value for blocked BUY zone (e.g., 50)';
COMMENT ON COLUMN smc_simple_pair_overrides.scalp_rsi_block_buy_max IS 'Maximum RSI value for blocked BUY zone (e.g., 60)';
COMMENT ON COLUMN smc_simple_pair_overrides.scalp_rsi_block_sell_min IS 'Minimum RSI value for blocked SELL zone';
COMMENT ON COLUMN smc_simple_pair_overrides.scalp_rsi_block_sell_max IS 'Maximum RSI value for blocked SELL zone';
COMMENT ON COLUMN smc_simple_pair_overrides.scalp_blocked_hours_utc IS 'Comma-separated UTC hours to block (e.g., "0,5,6,22,23")';

-- Example: Configure AUDUSD to block BUY when RSI 50-60 and block hours 0,5,6,22,23
-- UPDATE smc_simple_pair_overrides
-- SET scalp_rsi_block_buy_min = 50,
--     scalp_rsi_block_buy_max = 60,
--     scalp_blocked_hours_utc = '0,5,6,22,23'
-- WHERE epic = 'CS.D.AUDUSD.MINI.IP';

-- Verify columns were added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'smc_simple_pair_overrides'
AND column_name IN ('scalp_rsi_block_buy_min', 'scalp_rsi_block_buy_max',
                    'scalp_rsi_block_sell_min', 'scalp_rsi_block_sell_max',
                    'scalp_blocked_hours_utc')
ORDER BY column_name;
