-- Migration: Add partial close tracking columns to trade_log table
-- Created: 2025-10-23
-- Description: Adds columns to track partial position closes at break-even trigger

-- Add current_size column (tracks remaining position size)
ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS current_size FLOAT DEFAULT 1.0;

-- Add partial_close_executed column (flag indicating if partial close occurred)
ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS partial_close_executed BOOLEAN DEFAULT FALSE NOT NULL;

-- Add partial_close_time column (timestamp of partial close)
ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS partial_close_time TIMESTAMP NULL;

-- Update existing rows to have current_size = 1.0 if NULL
UPDATE trade_log
SET current_size = 1.0
WHERE current_size IS NULL;

-- Add comment for documentation
COMMENT ON COLUMN trade_log.current_size IS 'Remaining position size after partial closes (1.0 initially, 0.5 after 50% close)';
COMMENT ON COLUMN trade_log.partial_close_executed IS 'True if partial close was executed at break-even trigger';
COMMENT ON COLUMN trade_log.partial_close_time IS 'Timestamp when partial close occurred';

-- Verify the migration
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'trade_log'
AND column_name IN ('current_size', 'partial_close_executed', 'partial_close_time')
ORDER BY column_name;
