-- Migration: Add Dynamic VSL Trailing Columns
-- Date: 2026-01-14
-- Description: Adds columns to track dynamic VSL stage transitions for scalp trades
--
-- The dynamic VSL system moves the virtual stop loss to breakeven and beyond
-- as the trade moves into profit, minimizing losses on reversals.

-- Add dynamic VSL tracking columns to trade_log table
ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS vsl_stage VARCHAR(20) DEFAULT 'initial';

COMMENT ON COLUMN trade_log.vsl_stage IS
'Dynamic VSL stage: initial (fixed SL), breakeven (BE triggered), stage1 (profit locked)';

ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS vsl_breakeven_triggered BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN trade_log.vsl_breakeven_triggered IS
'True if the VSL moved to breakeven during this trade (profit reached +3 pips)';

ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS vsl_stage1_triggered BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN trade_log.vsl_stage1_triggered IS
'True if the VSL moved to stage1 during this trade (profit reached +4.5 pips, locking +2 pips)';

ALTER TABLE trade_log
ADD COLUMN IF NOT EXISTS vsl_peak_profit_pips FLOAT DEFAULT 0.0;

COMMENT ON COLUMN trade_log.vsl_peak_profit_pips IS
'Maximum favorable excursion (MFE) in pips - the peak profit reached before close';

-- Create index for analyzing dynamic VSL performance
CREATE INDEX IF NOT EXISTS idx_trade_log_vsl_stage
ON trade_log(vsl_stage)
WHERE is_scalp_trade = TRUE;

CREATE INDEX IF NOT EXISTS idx_trade_log_vsl_breakeven
ON trade_log(vsl_breakeven_triggered)
WHERE vsl_breakeven_triggered = TRUE;

-- Helpful view for analyzing dynamic VSL performance
CREATE OR REPLACE VIEW vsl_performance_analysis AS
SELECT
    DATE(timestamp) as trade_date,
    symbol,
    COUNT(*) as total_trades,
    SUM(CASE WHEN vsl_breakeven_triggered THEN 1 ELSE 0 END) as breakeven_triggered,
    SUM(CASE WHEN vsl_stage1_triggered THEN 1 ELSE 0 END) as stage1_triggered,
    SUM(CASE WHEN vsl_stage = 'initial' AND virtual_sl_triggered THEN 1 ELSE 0 END) as full_losses,
    ROUND(AVG(vsl_peak_profit_pips)::numeric, 2) as avg_peak_profit,
    ROUND(AVG(pips_gained)::numeric, 2) as avg_pips_gained,
    ROUND(
        100.0 * SUM(CASE WHEN vsl_breakeven_triggered THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        1
    ) as breakeven_rate_pct
FROM trade_log
WHERE is_scalp_trade = TRUE
  AND timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp), symbol
ORDER BY trade_date DESC, symbol;

COMMENT ON VIEW vsl_performance_analysis IS
'Analyze dynamic VSL performance: breakeven rate, stage triggers, average profits';
