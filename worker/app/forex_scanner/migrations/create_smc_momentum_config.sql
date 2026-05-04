-- ============================================================================
-- SMC_MOMENTUM Strategy Database Configuration
-- Liquidity Sweep + Rejection Wick — entry WITH HTF EMA50 trend direction
-- Gate 1 validated: PF 1.215 (5/5 pairs) with HTF alignment filter
-- ============================================================================
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /tmp/create_smc_momentum_config.sql

-- ============================================================================
-- 1. GLOBAL CONFIG TABLE (key/value rows, config_set scoped)
-- ============================================================================

CREATE TABLE IF NOT EXISTS smc_momentum_global_config (
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

-- ============================================================================
-- 2. DEFAULT PARAMETERS (demo config)
-- ============================================================================

INSERT INTO smc_momentum_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
VALUES
    -- Identity
    ('demo', 'strategy_name',          'SMC_MOMENTUM',   'string', 'general',  'Strategy identifier'),
    ('demo', 'version',                '1.0.0',          'string', 'general',  'Strategy version'),
    ('demo', 'is_active',              'false',          'bool',   'general',  'Master enable (false = no signals)'),

    -- Timeframes
    ('demo', 'htf_timeframe',          '4h',             'string', 'timeframes', 'HTF timeframe for EMA50 bias'),
    ('demo', 'entry_timeframe',        '15m',            'string', 'timeframes', 'Entry/sweep detection timeframe'),
    ('demo', 'atr_timeframe',          '1h',             'string', 'timeframes', 'ATR source timeframe'),

    -- Sweep detection
    ('demo', 'sweep_min_pips',         '3',              'float',  'sweep', 'Minimum sweep excess in pips (non-JPY)'),
    ('demo', 'sweep_max_pips',         '15',             'float',  'sweep', 'Maximum sweep excess in pips (non-JPY)'),
    ('demo', 'sweep_min_pips_jpy',     '3',              'float',  'sweep', 'Minimum sweep excess in pips (JPY)'),
    ('demo', 'sweep_max_pips_jpy',     '30',             'float',  'sweep', 'Maximum sweep excess in pips (JPY)'),

    -- Stop loss / take profit
    ('demo', 'sl_buffer_pips',         '5',              'float',  'risk', 'SL buffer beyond sweep candle high/low (non-JPY)'),
    ('demo', 'sl_buffer_pips_jpy',     '8',              'float',  'risk', 'SL buffer beyond sweep candle high/low (JPY)'),
    ('demo', 'tp_atr_multiplier',      '2.0',            'float',  'risk', 'TP = tp_atr_multiplier × ATR(atr_period, atr_timeframe)'),
    ('demo', 'atr_period',             '14',             'int',    'risk', 'ATR lookback period'),

    -- Confidence
    ('demo', 'min_confidence',         '0.55',           'float',  'confidence', 'Minimum confidence to emit signal'),
    ('demo', 'max_confidence',         '0.78',           'float',  'confidence', 'Confidence cap'),

    -- HTF alignment (Gate 1 confirmed load-bearing: PF 0.90→1.22)
    ('demo', 'htf_alignment_required', 'true',           'bool',   'filters', 'Require entry to be in HTF EMA50 trend direction'),
    ('demo', 'htf_ema_period',         '50',             'int',    'filters', 'EMA period for HTF trend determination'),

    -- Momentum quality filter
    ('demo', 'momentum_filter_mode',   'off',            'string', 'filters', 'off | volume | atr_expansion'),
    ('demo', 'volume_multiplier_threshold', '1.3',       'float',  'filters', 'Volume spike threshold (x 20-bar avg)'),
    ('demo', 'atr_expansion_threshold',    '1.3',        'float',  'filters', 'ATR expansion: current TR > threshold × rolling avg'),

    -- Swing pivot settings
    ('demo', 'swing_pivot_bars',       '2',              'int',    'sweep', 'Bars each side for 5-bar pivot detection'),
    ('demo', 'swing_max_age_bars',     '20',             'int',    'sweep', 'Max age (bars) for swing pivots to be considered'),

    -- Rollover block
    ('demo', 'rollover_block_enabled', 'true',           'bool',   'filters', 'Block entries during rollover window'),
    ('demo', 'rollover_start_hour',    '21',             'int',    'filters', 'Rollover block start (UTC hour, inclusive)'),
    ('demo', 'rollover_end_hour',      '23',             'int',    'filters', 'Rollover block end (UTC hour, exclusive)'),

    -- Cooldown
    ('demo', 'cooldown_minutes',       '240',            'int',    'cooldown', 'Cooldown between signals per pair (minutes)')

ON CONFLICT (config_set, parameter_name) DO NOTHING;

-- Live config mirrors demo defaults (adjust after validation)
INSERT INTO smc_momentum_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
SELECT 'live', parameter_name, parameter_value, value_type, category, description
FROM smc_momentum_global_config
WHERE config_set = 'demo'
ON CONFLICT (config_set, parameter_name) DO NOTHING;

-- ============================================================================
-- 3. PER-PAIR OVERRIDES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS smc_momentum_pair_overrides (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'demo',
    epic VARCHAR(60) NOT NULL,
    pair_name VARCHAR(10),

    -- Enablement
    is_enabled BOOLEAN DEFAULT FALSE,   -- disabled by default until Gate 2 passes
    is_traded BOOLEAN DEFAULT FALSE,
    monitor_only BOOLEAN DEFAULT TRUE,

    -- Per-pair scalar overrides (NULL = use global)
    min_confidence FLOAT,
    tp_atr_multiplier FLOAT,
    sl_buffer_pips FLOAT,
    sweep_min_pips FLOAT,
    sweep_max_pips FLOAT,
    htf_alignment_required BOOLEAN,
    momentum_filter_mode VARCHAR(20),
    cooldown_minutes INTEGER,

    -- JSONB ablation surface (ONLY surface for ablation — never set direct columns to sentinels)
    parameter_overrides JSONB DEFAULT '{}',

    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE (config_set, epic)
);

-- Phase 1 candidate pairs (enabled=true, monitor_only=true, is_traded=false)
INSERT INTO smc_momentum_pair_overrides (config_set, epic, pair_name, is_enabled, is_traded, monitor_only)
VALUES
    ('demo', 'CS.D.USDJPY.MINI.IP', 'USDJPY', TRUE,  FALSE, TRUE),
    ('demo', 'CS.D.EURJPY.MINI.IP', 'EURJPY', TRUE,  FALSE, TRUE),
    ('demo', 'CS.D.AUDJPY.MINI.IP', 'AUDJPY', TRUE,  FALSE, TRUE),
    ('demo', 'CS.D.NZDUSD.MINI.IP', 'NZDUSD', FALSE, FALSE, TRUE),  -- borderline Gate 1 (PF 1.02), disabled pending Gate 2
    ('demo', 'CS.D.AUDUSD.MINI.IP', 'AUDUSD', TRUE,  FALSE, TRUE),
    -- Disabled pairs (can re-enable after Phase 1 validates)
    ('demo', 'CS.D.EURUSD.CEEM.IP', 'EURUSD', FALSE, FALSE, TRUE),
    ('demo', 'CS.D.GBPUSD.MINI.IP', 'GBPUSD', FALSE, FALSE, TRUE),
    ('demo', 'CS.D.USDCHF.MINI.IP', 'USDCHF', FALSE, FALSE, TRUE),
    ('demo', 'CS.D.USDCAD.MINI.IP', 'USDCAD', FALSE, FALSE, TRUE)
ON CONFLICT (config_set, epic) DO NOTHING;

-- Mirror to live (all disabled until live promotion)
INSERT INTO smc_momentum_pair_overrides (config_set, epic, pair_name, is_enabled, is_traded, monitor_only)
SELECT 'live', epic, pair_name, FALSE, FALSE, TRUE
FROM smc_momentum_pair_overrides
WHERE config_set = 'demo'
ON CONFLICT (config_set, epic) DO NOTHING;

-- ============================================================================
-- 4. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_smc_momentum_global_config_set
    ON smc_momentum_global_config (config_set, is_active);

CREATE INDEX IF NOT EXISTS idx_smc_momentum_pair_overrides_config_set
    ON smc_momentum_pair_overrides (config_set, is_enabled);

-- ============================================================================
-- 5. VERIFICATION
-- ============================================================================

SELECT 'SMC_MOMENTUM tables created' AS status;
SELECT config_set, parameter_name, parameter_value
FROM smc_momentum_global_config
WHERE config_set = 'demo'
ORDER BY category, parameter_name;
SELECT config_set, epic, pair_name, is_enabled, monitor_only
FROM smc_momentum_pair_overrides
WHERE config_set = 'demo'
ORDER BY pair_name;
