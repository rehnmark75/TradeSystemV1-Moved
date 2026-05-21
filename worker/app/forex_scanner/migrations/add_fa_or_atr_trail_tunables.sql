-- Add FA_OR_ATR_TRAIL tunables to strategy_config after initial enablement.
-- Run against: strategy_config

INSERT INTO fa_or_atr_trail_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
VALUES
    ('demo', 'fa_or_session_start_hour', '8', 'int', 'session', 'UTC session start hour'),
    ('demo', 'fa_or_session_end_hour', '20', 'int', 'session', 'UTC session end hour'),
    ('demo', 'fa_or_adx_min', '18.0', 'float', 'filters', 'Minimum ADX for trend participation'),
    ('demo', 'fa_or_min_slope_pips', '0.3', 'float', 'filters', 'Minimum EMA slope in pips'),
    ('demo', 'fa_or_max_vwap_atr', '3.0', 'float', 'filters', 'Maximum close-to-VWAP distance in ATR units'),
    ('demo', 'fa_or_opening_range_bars', '6', 'int', 'entry', 'Opening range bars'),
    ('demo', 'fa_or_value_area_std', '0.70', 'float', 'entry', 'Value-area standard deviation multiplier'),
    ('demo', 'fa_or_vp_lookback', '15', 'int', 'entry', 'Value-area lookback bars'),
    ('demo', 'fa_or_cooldown_bars', '5', 'int', 'cooldown', 'Minimum bars between signals per pair'),
    ('demo', 'fa_or_sl_atr', '1.2', 'float', 'risk', 'Stop loss ATR multiplier'),
    ('demo', 'fa_or_tp_atr', '2.0', 'float', 'risk', 'Take profit ATR multiplier'),
    ('demo', 'fa_or_trail_trigger_atr', '0.25', 'float', 'risk', 'Trailing stop activation in ATR units'),
    ('demo', 'fa_or_trail_distance_atr', '0.10', 'float', 'risk', 'Trailing stop distance in ATR units'),
    ('demo', 'fa_or_atr_period', '14', 'int', 'indicators', 'ATR period'),
    ('demo', 'fa_or_adx_period', '14', 'int', 'indicators', 'ADX period'),
    ('demo', 'fa_or_rsi_period', '14', 'int', 'indicators', 'RSI period'),
    ('demo', 'fa_or_htf_ema_period', '50', 'int', 'indicators', 'Higher timeframe EMA period'),
    ('demo', 'fa_or_usdjpy_atr_floor_pips', '8.7', 'float', 'usd_jpy_filter', 'USDJPY-only minimum ATR floor in pips; ignored by non-USDJPY epics')
ON CONFLICT (config_set, parameter_name) DO UPDATE SET
    parameter_value = EXCLUDED.parameter_value,
    value_type = EXCLUDED.value_type,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    is_active = TRUE,
    updated_at = NOW();

INSERT INTO fa_or_atr_trail_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
SELECT 'live', parameter_name, parameter_value, value_type, category, description
FROM fa_or_atr_trail_global_config
WHERE config_set = 'demo'
ON CONFLICT (config_set, parameter_name) DO NOTHING;

ALTER TABLE fa_or_atr_trail_pair_overrides
    ADD COLUMN IF NOT EXISTS fa_or_session_start_hour INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_session_end_hour INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_adx_min FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_min_slope_pips FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_max_vwap_atr FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_opening_range_bars INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_value_area_std FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_vp_lookback INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_cooldown_bars INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_sl_atr FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_tp_atr FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_trail_trigger_atr FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_trail_distance_atr FLOAT,
    ADD COLUMN IF NOT EXISTS fa_or_atr_period INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_adx_period INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_rsi_period INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_htf_ema_period INTEGER,
    ADD COLUMN IF NOT EXISTS fa_or_usdjpy_atr_floor_pips FLOAT;
