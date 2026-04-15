-- ============================================================================
-- XAU_GOLD Strategy Database Configuration
-- ============================================================================
-- Gold-optimized multi-timeframe SMC-style strategy.
--   Tier 1: 4H EMA(50/200) bias
--   Tier 2: 1H swing break / BOS with MACD confirmation
--   Tier 3: 15m pullback to OB / FVG / fib zone
--
-- Apply:
--   docker exec postgres psql -U postgres -d strategy_config \
--     -f /app/forex_scanner/migrations/create_xau_gold_config.sql
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. GLOBAL CONFIG (key/value)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS xau_gold_global_config (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'live',
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',  -- string|int|float|bool|json
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    display_order INTEGER DEFAULT 0,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_editable BOOLEAN DEFAULT TRUE,
    updated_by VARCHAR(100),
    change_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO xau_gold_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description, display_order)
VALUES
    -- Identity
    ('live', 'strategy_name',          'XAU_GOLD',         'string', 'general', 'Strategy name',            1),
    ('live', 'version',                '1.0.0',            'string', 'general', 'Strategy version',         2),
    ('live', 'enabled',                'true',             'bool',   'general', 'Enable/disable strategy',  3),

    -- Timeframes
    ('live', 'htf_timeframe',          '4h',               'string', 'timeframes', 'HTF bias timeframe',   10),
    ('live', 'trigger_timeframe',      '1h',               'string', 'timeframes', 'BOS trigger timeframe',11),
    ('live', 'entry_timeframe',        '15m',              'string', 'timeframes', 'Entry timeframe',      12),

    -- Core indicators
    ('live', 'ema_fast_period',        '50',               'int',    'indicators', 'Fast EMA on HTF',      20),
    ('live', 'ema_slow_period',        '100',              'int',    'indicators', 'Slow EMA on HTF',      21),
    ('live', 'atr_period',             '14',               'int',    'indicators', 'ATR period',           22),
    ('live', 'rsi_period',             '14',               'int',    'indicators', 'RSI period',           23),
    ('live', 'adx_period',             '14',               'int',    'indicators', 'ADX period',           24),
    ('live', 'swing_lookback',         '20',               'int',    'indicators', 'Swing lookback bars',  25),

    -- Confidence
    ('live', 'min_confidence',         '0.58',             'float',  'confidence', 'Min confidence',       30),
    ('live', 'max_confidence',         '0.75',             'float',  'confidence', 'Max confidence cap (paradox)', 31),
    ('live', 'base_confidence',        '0.50',             'float',  'confidence', 'Base confidence',      32),

    -- Confluence weights (sum cap at 0.25 above base)
    ('live', 'w_htf_bias',             '0.08',             'float',  'confluence', '4H bias alignment',    40),
    ('live', 'w_bos_displacement',     '0.08',             'float',  'confluence', '1H BOS with displacement', 41),
    ('live', 'w_entry_pullback',       '0.06',             'float',  'confluence', '15m entry at untested level', 42),
    ('live', 'w_dxy_confluence',       '0.05',             'float',  'confluence', 'DXY inverse confirmation', 43),
    ('live', 'w_rsi_neutral',          '0.04',             'float',  'confluence', 'RSI 40-60 neutral band', 44),

    -- Risk: SL/TP
    ('live', 'sl_atr_multiplier',      '1.5',              'float',  'risk', 'SL as ATR multiple',         50),
    ('live', 'min_stop_loss_pips',     '25.0',             'float',  'risk', 'Floor for SL',               51),
    ('live', 'max_stop_loss_pips',     '80.0',             'float',  'risk', 'Cap for SL',                 52),
    ('live', 'rr_ratio',               '2.0',              'float',  'risk', 'Target R:R',                 53),
    ('live', 'min_rr_ratio',           '1.33',             'float',  'risk', 'Hard floor for R:R',         54),
    ('live', 'min_tp_pips',            '15.0',             'float',  'risk', 'Floor for TP',               55),
    ('live', 'fixed_sl_tp_override_enabled', 'false',      'bool',   'risk', 'Use fixed SL/TP instead of ATR', 56),
    ('live', 'fixed_stop_loss_pips',   '40.0',             'float',  'risk', 'Fixed SL fallback',          57),
    ('live', 'fixed_take_profit_pips', '80.0',             'float',  'risk', 'Fixed TP fallback',          58),

    -- Regime gating
    ('live', 'adx_trending_threshold', '25.0',             'float',  'regime', 'ADX above = trending',     60),
    ('live', 'adx_ranging_threshold',  '20.0',             'float',  'regime', 'ADX below = ranging (blocked)', 61),
    ('live', 'atr_expansion_pct',      '85.0',             'float',  'regime', 'ATR percentile = expansion (blocked)', 62),
    ('live', 'atr_pct_lookback_bars',  '120',              'int',    'regime', 'ATR percentile lookback (4H bars)', 63),
    ('live', 'block_ranging',          'true',             'bool',   'regime', 'Block signals in ranging regime', 64),
    ('live', 'block_expansion',        'true',             'bool',   'regime', 'Block signals in news/expansion', 65),

    -- Session filter (UTC hours, inclusive start, exclusive end)
    ('live', 'session_filter_enabled', 'true',             'bool',   'session', 'Enable session filter',   70),
    ('live', 'london_start_hour',      '7',                'int',    'session', 'London session start UTC', 71),
    ('live', 'london_end_hour',        '10',               'int',    'session', 'London session end UTC',   72),
    ('live', 'ny_start_hour',          '13',               'int',    'session', 'NY session start UTC',     73),
    ('live', 'ny_end_hour',            '20',               'int',    'session', 'NY session end UTC',       74),
    ('live', 'rollover_start_hour',    '21',               'int',    'session', 'Rollover start (blocked)', 75),
    ('live', 'rollover_end_hour',      '22',               'int',    'session', 'Rollover end (blocked)',   76),
    ('live', 'asian_allowed',          'false',            'bool',   'session', 'Allow Asian session signals', 77),

    -- Structure / entry
    ('live', 'bos_displacement_atr_mult', '1.2',           'float',  'structure', '1H BOS candle displacement vs ATR', 80),
    ('live', 'fib_pullback_min',       '0.382',            'float',  'structure', 'Min pullback depth',   81),
    ('live', 'fib_pullback_max',       '0.618',            'float',  'structure', 'Max pullback depth',   82),
    ('live', 'bos_expiry_hours',       '12',               'float',  'structure', 'Max BOS age in hours', 83),
    ('live', 'bos_search_bars',        '24',               'int',    'structure', 'Recent 1H bars to search for BOS', 84),
    ('live', 'entry_check_bars',       '12',               'int',    'structure', 'Recent 15m bars to allow pullback entry', 85),
    ('live', 'require_ob_or_fvg',      'true',             'bool',   'structure', 'Require untested OB/FVG overlap', 86),

    -- Cooldown / limits
    ('live', 'signal_cooldown_minutes','180',              'int',    'limits', 'Cooldown per epic',       90),
    ('live', 'max_concurrent_signals', '1',                'int',    'limits', 'Max concurrent signals',  91),

    -- Filters
    ('live', 'macd_filter_enabled',    'true',             'bool',   'filters', 'MACD alignment on 1H',   100),
    ('live', 'dxy_confluence_enabled', 'true',             'bool',   'filters', 'Check DXY inverse correlation', 101),
    ('live', 'rsi_neutral_min',        '40.0',             'float',  'filters', 'RSI lower bound for neutral', 102),
    ('live', 'rsi_neutral_max',        '60.0',             'float',  'filters', 'RSI upper bound for neutral', 103),

    -- Enabled pairs (JSON array of epics)
    ('live', 'enabled_pairs',          '["CS.D.CFEGOLD.CEE.IP"]', 'json', 'pairs', 'Enabled gold epics', 110)
ON CONFLICT (config_set, parameter_name) DO NOTHING;

INSERT INTO xau_gold_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description, display_order)
SELECT
    'demo', parameter_name, parameter_value, value_type, category, description, display_order
FROM xau_gold_global_config
WHERE config_set = 'live'
ON CONFLICT (config_set, parameter_name) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 2. PER-PAIR OVERRIDES
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS xau_gold_pair_overrides (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'live',
    epic VARCHAR(50) NOT NULL,
    pair_name VARCHAR(16),
    pip_size FLOAT NOT NULL DEFAULT 0.1,  -- XAUUSD: 0.1 per pip

    -- Overrides (NULL = use global)
    fixed_stop_loss_pips FLOAT,
    fixed_take_profit_pips FLOAT,
    min_confidence FLOAT,
    max_confidence FLOAT,
    sl_atr_multiplier FLOAT,
    rr_ratio FLOAT,
    signal_cooldown_minutes INTEGER,

    -- JSONB bag for arbitrary overrides
    parameter_overrides JSONB DEFAULT '{}'::jsonb,

    -- Flags
    is_enabled BOOLEAN DEFAULT TRUE,
    is_traded BOOLEAN DEFAULT FALSE,  -- gated off by default (backtest-first)
    monitor_only BOOLEAN DEFAULT TRUE, -- signals logged only, no execution

    notes TEXT,
    updated_by VARCHAR(100),
    change_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO xau_gold_pair_overrides (config_set, epic, pair_name, pip_size, is_enabled, is_traded, monitor_only, notes)
VALUES
    ('live', 'CS.D.CFEGOLD.CEE.IP', 'XAUUSD', 0.1, TRUE, FALSE, TRUE,
     'Backtest-first rollout. IG may flag otcTradeable:false; verify before enabling is_traded.')
ON CONFLICT (config_set, epic) DO NOTHING;

INSERT INTO xau_gold_pair_overrides (config_set, epic, pair_name, pip_size, is_enabled, is_traded, monitor_only, notes)
SELECT
    'demo', epic, pair_name, pip_size, is_enabled, is_traded, monitor_only, notes
FROM xau_gold_pair_overrides
WHERE config_set = 'live'
ON CONFLICT (config_set, epic) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 3. AUDIT
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS xau_gold_config_audit (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'live',
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    parameter_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    change_type VARCHAR(20) NOT NULL,
    changed_by VARCHAR(100) DEFAULT 'system',
    change_reason TEXT,
    changed_at TIMESTAMPTZ DEFAULT NOW()
);


-- ----------------------------------------------------------------------------
-- 4. INDEXES
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_xau_gold_global_category ON xau_gold_global_config(category);
CREATE INDEX IF NOT EXISTS idx_xau_gold_global_active   ON xau_gold_global_config(is_active);
CREATE INDEX IF NOT EXISTS idx_xau_gold_pair_enabled    ON xau_gold_pair_overrides(is_enabled);
CREATE UNIQUE INDEX IF NOT EXISTS idx_xau_gold_global_scope_param ON xau_gold_global_config(config_set, parameter_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_xau_gold_pair_scope_epic    ON xau_gold_pair_overrides(config_set, epic);


SELECT 'XAU_GOLD configuration tables created' AS status;
