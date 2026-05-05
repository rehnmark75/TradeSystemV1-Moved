-- Add initial_sl_price to capture the SL at order placement (before any trailing moves it)
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS initial_sl_price DOUBLE PRECISION;

-- Backfill: trades where stop was never moved still have the original SL in sl_price
UPDATE trade_log
SET initial_sl_price = sl_price
WHERE stop_limit_changes_count = 0
  AND sl_price IS NOT NULL
  AND initial_sl_price IS NULL;
