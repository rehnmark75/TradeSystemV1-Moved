-- Migration: Add scalp_use_limit_orders config option
-- Date: 2026-01-19
-- Version: 3.3.0
-- Purpose: Enable LIMIT order type (better price entry) instead of STOP order type (momentum confirmation) for scalp trades
--
-- LIMIT orders: Entry when price pulls back to level (better entry price, lower fill rate)
-- STOP orders: Entry when price breaks through level (momentum confirmation, worse entry price)
--
-- Based on trade analysis showing 5.13 pip average slippage with STOP orders,
-- which is devastating when average wins are only 2.61 pips.

-- Connect to strategy_config database
\c strategy_config;

-- Add scalp_use_limit_orders column to smc_simple_global_config
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_use_limit_orders BOOLEAN DEFAULT FALSE;

-- Update comment to document the field
COMMENT ON COLUMN smc_simple_global_config.scalp_use_limit_orders IS
'v3.3.0: Use LIMIT orders (better price entry) instead of STOP orders (momentum confirmation) for scalp trades. LIMIT orders enter when price pulls back to entry level, giving better entry prices but lower fill rates.';

-- Log the migration
INSERT INTO smc_simple_config_audit (
    config_type,
    field_name,
    old_value,
    new_value,
    change_reason,
    changed_by
)
VALUES (
    'global',
    'scalp_use_limit_orders',
    NULL,
    'FALSE (column added)',
    'v3.3.0: Add scalp_use_limit_orders config to switch between LIMIT (better price) and STOP (momentum) order types',
    'migration'
);

-- Add metadata for the new field
INSERT INTO smc_simple_parameter_metadata (
    parameter_name,
    display_name,
    description,
    category,
    data_type,
    default_value,
    min_value,
    max_value,
    is_advanced,
    sort_order
)
VALUES (
    'scalp_use_limit_orders',
    'Scalp Use Limit Orders',
    'Use LIMIT orders (better entry price, lower fill rate) instead of STOP orders (momentum confirmation, worse entry price) for scalp trades. Based on analysis showing 5+ pip slippage with STOP orders.',
    'scalp_mode',
    'boolean',
    'false',
    NULL,
    NULL,
    TRUE,
    256
)
ON CONFLICT (parameter_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    category = EXCLUDED.category;

SELECT 'Migration completed: scalp_use_limit_orders column added' AS status;
