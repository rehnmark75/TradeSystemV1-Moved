
-- Fix moved_to_breakeven for trades that actually moved to breakeven
-- These are trades where the stop loss is above entry (BUY) or below entry (SELL)

UPDATE trade_log
SET moved_to_breakeven = true
WHERE moved_to_breakeven = false
AND (
    (direction = 'BUY' AND sl_price >= entry_price) OR
    (direction = 'SELL' AND sl_price <= entry_price)
)
AND status IN ('closed', 'expired', 'break_even', 'trailing')
AND timestamp >= '2025-08-12'  -- After the bug started
;

-- Report on what was fixed
SELECT
    'Fixed moved_to_breakeven flags' as action,
    COUNT(*) as trades_affected
FROM trade_log
WHERE moved_to_breakeven = true
AND (
    (direction = 'BUY' AND sl_price >= entry_price) OR
    (direction = 'SELL' AND sl_price <= entry_price)
)
AND timestamp >= '2025-08-12';
