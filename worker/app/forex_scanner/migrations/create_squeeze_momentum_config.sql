-- ============================================================================
-- SQUEEZE_MOMENTUM Strategy Database Configuration
-- LazyBear-style BB/KC squeeze release on 15m with 1H EMA50 alignment.
-- Monitor-only defaults for theory testing.
-- ============================================================================
-- Run:
-- docker cp worker/app/forex_scanner/migrations/create_squeeze_momentum_config.sql postgres:/tmp/create_squeeze_momentum_config.sql
-- docker exec postgres psql -U postgres -d strategy_config -f /tmp/create_squeeze_momentum_config.sql

CREATE TABLE IF NOT EXISTS squeeze_momentum_global_config (
    id SERIAL PRIMARY KEY,

    strategy_name VARCHAR(50) NOT NULL DEFAULT 'SQUEEZE_MOMENTUM',
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    entry_timeframe VARCHAR(10) NOT NULL DEFAULT '15m',
    htf_timeframe VARCHAR(10) NOT NULL DEFAULT '1h',

    bb_length INTEGER NOT NULL DEFAULT 20,
    bb_mult FLOAT NOT NULL DEFAULT 2.0,
    kc_length INTEGER NOT NULL DEFAULT 20,
    kc_mult FLOAT NOT NULL DEFAULT 1.5,
    use_true_range BOOLEAN NOT NULL DEFAULT TRUE,

    htf_ema_period INTEGER NOT NULL DEFAULT 50,
    adx_period INTEGER NOT NULL DEFAULT 14,
    adx_min FLOAT NOT NULL DEFAULT 18.0,
    require_adx_rising BOOLEAN NOT NULL DEFAULT TRUE,

    squeeze_min_bars INTEGER NOT NULL DEFAULT 3,
    squeeze_lookback_bars INTEGER NOT NULL DEFAULT 8,
    require_release_bar BOOLEAN NOT NULL DEFAULT TRUE,
    min_momentum_slope_atr FLOAT NOT NULL DEFAULT 0.03,

    atr_period INTEGER NOT NULL DEFAULT 14,
    stop_atr_multiplier FLOAT NOT NULL DEFAULT 1.5,
    take_profit_atr_multiplier FLOAT NOT NULL DEFAULT 2.5,

    min_confidence FLOAT NOT NULL DEFAULT 0.55,
    max_confidence FLOAT NOT NULL DEFAULT 0.88,
    signal_cooldown_minutes INTEGER NOT NULL DEFAULT 180,
    block_asian_session BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO squeeze_momentum_global_config DEFAULT VALUES
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS squeeze_momentum_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(60) NOT NULL UNIQUE,
    pair_name VARCHAR(10),

    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    is_traded BOOLEAN NOT NULL DEFAULT FALSE,
    monitor_only BOOLEAN NOT NULL DEFAULT TRUE,

    signal_cooldown_minutes INTEGER,
    min_momentum_slope_atr FLOAT,
    adx_min FLOAT,
    stop_atr_multiplier FLOAT,
    take_profit_atr_multiplier FLOAT,
    min_confidence FLOAT,
    max_confidence FLOAT,

    parameter_overrides JSONB NOT NULL DEFAULT '{}',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO squeeze_momentum_pair_overrides
    (epic, pair_name, is_enabled, is_traded, monitor_only, notes)
VALUES
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.GBPUSD.MINI.IP', 'GBPUSD', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.AUDUSD.MINI.IP', 'AUDUSD', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.NZDUSD.MINI.IP', 'NZDUSD', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.USDCAD.MINI.IP', 'USDCAD', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.USDCHF.MINI.IP', 'USDCHF', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.USDJPY.MINI.IP', 'USDJPY', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.EURJPY.MINI.IP', 'EURJPY', TRUE, FALSE, TRUE, 'Monitor-only theory test'),
    ('CS.D.AUDJPY.MINI.IP', 'AUDJPY', TRUE, FALSE, TRUE, 'Monitor-only theory test')
ON CONFLICT (epic) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_squeeze_momentum_global_active
    ON squeeze_momentum_global_config (is_active);

CREATE INDEX IF NOT EXISTS idx_squeeze_momentum_pair_enabled
    ON squeeze_momentum_pair_overrides (is_enabled);

SELECT 'SQUEEZE_MOMENTUM tables created' AS status;

SELECT id, version, entry_timeframe, htf_timeframe, bb_length, bb_mult,
       kc_length, kc_mult, squeeze_min_bars, squeeze_lookback_bars,
       min_momentum_slope_atr, stop_atr_multiplier, take_profit_atr_multiplier
FROM squeeze_momentum_global_config
WHERE is_active = TRUE;

SELECT epic, pair_name, is_enabled, monitor_only
FROM squeeze_momentum_pair_overrides
ORDER BY pair_name;
