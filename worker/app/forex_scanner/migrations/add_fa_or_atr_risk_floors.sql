-- Add optional fixed pip floors for FA_OR_ATR_TRAIL risk sizing.
-- When set per pair, the strategy still uses ATR sizing whenever ATR is wider.

INSERT INTO fa_or_atr_trail_global_config (
    config_set,
    parameter_name,
    parameter_value,
    value_type,
    category,
    description,
    is_active
)
VALUES
    ('demo', 'fa_or_min_sl_pips', '0.0', 'float', 'risk', 'Minimum stop loss distance in pips; 0 disables the floor.', TRUE),
    ('demo', 'fa_or_min_tp_pips', '0.0', 'float', 'risk', 'Minimum take profit distance in pips; 0 disables the floor.', TRUE),
    ('live', 'fa_or_min_sl_pips', '0.0', 'float', 'risk', 'Minimum stop loss distance in pips; 0 disables the floor.', TRUE),
    ('live', 'fa_or_min_tp_pips', '0.0', 'float', 'risk', 'Minimum take profit distance in pips; 0 disables the floor.', TRUE)
ON CONFLICT (config_set, parameter_name) DO NOTHING;

ALTER TABLE fa_or_atr_trail_pair_overrides
    ADD COLUMN IF NOT EXISTS fa_or_min_sl_pips FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_min_tp_pips FLOAT;

UPDATE fa_or_atr_trail_pair_overrides fa
SET
    fa_or_min_sl_pips = rf.fixed_stop_loss_pips,
    fa_or_min_tp_pips = rf.fixed_take_profit_pips,
    parameter_overrides = COALESCE(fa.parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'fa_or_min_sl_pips', rf.fixed_stop_loss_pips,
            'fa_or_min_tp_pips', rf.fixed_take_profit_pips
        ),
    updated_at = NOW()
FROM range_fade_pair_overrides rf
WHERE fa.config_set = rf.config_set
    AND fa.epic = rf.epic
    AND fa.is_enabled IS TRUE
    AND rf.fixed_stop_loss_pips IS NOT NULL
    AND rf.fixed_take_profit_pips IS NOT NULL;
