-- ============================================================================
-- ADD FIXED SL/TP COLUMNS FOR PER-PAIR OVERRIDE
-- ============================================================================
-- Database: strategy_config
-- Purpose: Add per-pair stop loss and take profit override columns
-- Date: 2026-01-02
-- ============================================================================

-- Add to global config: master switch for fixed SL/TP mode
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS fixed_sl_tp_override_enabled BOOLEAN DEFAULT TRUE;

-- Add default fixed values to global config (used when pair-specific not set)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS fixed_stop_loss_pips NUMERIC(5,1) DEFAULT 9.0;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS fixed_take_profit_pips NUMERIC(5,1) DEFAULT 15.0;

-- Add per-pair SL/TP overrides
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS fixed_stop_loss_pips NUMERIC(5,1);

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS fixed_take_profit_pips NUMERIC(5,1);

-- Add comments
COMMENT ON COLUMN smc_simple_global_config.fixed_sl_tp_override_enabled IS 'Master switch: when TRUE, use fixed SL/TP values instead of strategy-calculated';
COMMENT ON COLUMN smc_simple_global_config.fixed_stop_loss_pips IS 'Default fixed stop loss in pips (when override enabled and no pair-specific value)';
COMMENT ON COLUMN smc_simple_global_config.fixed_take_profit_pips IS 'Default fixed take profit in pips (when override enabled and no pair-specific value)';
COMMENT ON COLUMN smc_simple_pair_overrides.fixed_stop_loss_pips IS 'Per-pair fixed stop loss override in pips (NULL = use global default)';
COMMENT ON COLUMN smc_simple_pair_overrides.fixed_take_profit_pips IS 'Per-pair fixed take profit override in pips (NULL = use global default)';

-- Update the current active config with values from config.py
UPDATE smc_simple_global_config
SET
    fixed_sl_tp_override_enabled = TRUE,
    fixed_stop_loss_pips = 9.0,
    fixed_take_profit_pips = 15.0
WHERE is_active = TRUE;

-- ============================================================================
-- EXAMPLE: Set per-pair SL/TP values (uncomment and modify as needed)
-- ============================================================================
-- UPDATE smc_simple_pair_overrides SET fixed_stop_loss_pips = 12, fixed_take_profit_pips = 20 WHERE epic = 'CS.D.USDJPY.MINI.IP';
-- UPDATE smc_simple_pair_overrides SET fixed_stop_loss_pips = 10, fixed_take_profit_pips = 18 WHERE epic = 'CS.D.GBPUSD.MINI.IP';
