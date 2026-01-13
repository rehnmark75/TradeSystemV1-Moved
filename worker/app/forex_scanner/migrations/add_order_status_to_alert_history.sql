-- Migration: add_order_status_to_alert_history.sql
-- Date: 2026-01-13
-- Purpose: Track order lifecycle for status-based cooldowns
--
-- This enables intelligent cooldown management:
-- - Full 4h cooldown only for FILLED trades
-- - 30min cooldown for expired/placed/pending orders
-- - 15min cooldown for rejected orders
--
-- Status values:
--   pending  - Signal generated, order not yet sent
--   placed   - Order sent to broker (working order)
--   filled   - Order executed, trade opened
--   expired  - Limit order expired without filling
--   rejected - Order rejected by broker

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS order_status VARCHAR(20) DEFAULT 'pending';

-- Create index for efficient cooldown queries
CREATE INDEX IF NOT EXISTS idx_alert_history_order_status
ON alert_history(epic, order_status, alert_timestamp);

-- Add comment for documentation
COMMENT ON COLUMN alert_history.order_status IS
'Order lifecycle status: pending->placed->filled/expired/rejected. Used for status-based cooldowns.';

-- Backfill existing records: assume all past alerts resulted in placed orders
-- (conservative estimate - they went through execution)
UPDATE alert_history
SET order_status = 'placed'
WHERE order_status IS NULL OR order_status = 'pending';
