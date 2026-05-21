-- FA_OR_ATR_TRAIL demo/live strategy configuration.
-- Run against: strategy_config

CREATE TABLE IF NOT EXISTS fa_or_atr_trail_global_config (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'demo',
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (config_set, parameter_name)
);

INSERT INTO fa_or_atr_trail_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
VALUES
    ('demo', 'strategy_name', 'FA_OR_ATR_TRAIL', 'string', 'general', 'Strategy identifier'),
    ('demo', 'version', '0.1.0', 'string', 'general', 'Strategy version'),
    ('demo', 'is_active', 'true', 'bool', 'general', 'Master enable for demo forward testing'),
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
SELECT 'live', parameter_name,
       CASE WHEN parameter_name = 'is_active' THEN 'false' ELSE parameter_value END,
       value_type, category, description
FROM fa_or_atr_trail_global_config
WHERE config_set = 'demo'
ON CONFLICT (config_set, parameter_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS fa_or_atr_trail_pair_overrides (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'demo',
    epic VARCHAR(60) NOT NULL,
    pair_name VARCHAR(10),
    is_enabled BOOLEAN DEFAULT FALSE,
    is_traded BOOLEAN DEFAULT FALSE,
    monitor_only BOOLEAN DEFAULT TRUE,
    fa_or_session_start_hour INTEGER,
    fa_or_session_end_hour INTEGER,
    fa_or_adx_min FLOAT,
    fa_or_min_slope_pips FLOAT,
    fa_or_max_vwap_atr FLOAT,
    fa_or_opening_range_bars INTEGER,
    fa_or_value_area_std FLOAT,
    fa_or_vp_lookback INTEGER,
    fa_or_cooldown_bars INTEGER,
    fa_or_sl_atr FLOAT,
    fa_or_tp_atr FLOAT,
    fa_or_trail_trigger_atr FLOAT,
    fa_or_trail_distance_atr FLOAT,
    fa_or_atr_period INTEGER,
    fa_or_adx_period INTEGER,
    fa_or_rsi_period INTEGER,
    fa_or_htf_ema_period INTEGER,
    fa_or_usdjpy_atr_floor_pips FLOAT,
    parameter_overrides JSONB DEFAULT '{}',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (config_set, epic)
);

INSERT INTO fa_or_atr_trail_pair_overrides
    (config_set, epic, pair_name, is_enabled, is_traded, monitor_only, notes)
VALUES
    ('demo', 'CS.D.USDJPY.MINI.IP', 'USDJPY', TRUE, TRUE, FALSE, 'Forward-test active; 30d PF cleared target with USDJPY ATR floor.'),
    ('demo', 'CS.D.EURUSD.CEEM.IP', 'EURUSD', TRUE, FALSE, TRUE, 'Monitoring only; 30d PF below target.'),
    ('demo', 'CS.D.GBPUSD.MINI.IP', 'GBPUSD', TRUE, FALSE, TRUE, 'Monitoring only; 30d PF below target.'),
    ('demo', 'CS.D.AUDUSD.MINI.IP', 'AUDUSD', TRUE, FALSE, TRUE, 'Monitoring only; 30d PF below target.'),
    ('demo', 'CS.D.USDCHF.MINI.IP', 'USDCHF', TRUE, FALSE, TRUE, 'Monitoring only; 30d PF below target.'),
    ('demo', 'CS.D.EURJPY.MINI.IP', 'EURJPY', TRUE, FALSE, TRUE, 'Monitoring only; 30d PF below target.'),
    ('demo', 'CS.D.AUDJPY.MINI.IP', 'AUDJPY', TRUE, FALSE, TRUE, 'Monitoring only; 30d PF below target.')
ON CONFLICT (config_set, epic) DO UPDATE SET
    pair_name = EXCLUDED.pair_name,
    is_enabled = EXCLUDED.is_enabled,
    is_traded = EXCLUDED.is_traded,
    monitor_only = EXCLUDED.monitor_only,
    notes = EXCLUDED.notes,
    updated_at = NOW();

INSERT INTO fa_or_atr_trail_pair_overrides
    (config_set, epic, pair_name, is_enabled, is_traded, monitor_only, notes)
SELECT 'live', epic, pair_name, FALSE, FALSE, TRUE, 'Disabled in live until demo forward-test promotion.'
FROM fa_or_atr_trail_pair_overrides
WHERE config_set = 'demo'
ON CONFLICT (config_set, epic) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_fa_or_atr_trail_pair_enabled
    ON fa_or_atr_trail_pair_overrides (config_set, is_enabled);

INSERT INTO enabled_strategies
    (strategy_name, is_enabled, is_backtest_only, display_name, description, strategy_type)
VALUES
    ('FA_OR_ATR_TRAIL', TRUE, FALSE, 'FA OR ATR Trail', 'Failed-auction/opening-range ATR trailing strategy', 'signal')
ON CONFLICT (strategy_name) DO UPDATE SET
    is_enabled = TRUE,
    is_backtest_only = FALSE,
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    strategy_type = EXCLUDED.strategy_type,
    updated_at = NOW();

UPDATE scanner_global_config
SET enabled_strategies =
    CASE
        WHEN enabled_strategies IS NULL THEN '["FA_OR_ATR_TRAIL"]'::jsonb
        WHEN enabled_strategies @> '["FA_OR_ATR_TRAIL"]'::jsonb THEN enabled_strategies
        ELSE enabled_strategies || '["FA_OR_ATR_TRAIL"]'::jsonb
    END,
    updated_at = NOW()
WHERE config_set = 'demo' AND is_active = TRUE;
