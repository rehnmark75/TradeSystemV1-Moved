-- ============================================================================
-- IMPULSE_FADE Strategy Database Configuration
-- Fade large 5m candle bodies (body >= N×ATR14) during late-US session window.
-- Direction is always against the impulse: SHORT on bullish, LONG on bearish.
-- Inverted R:R (TP=8, SL=12) compensated by high WR edge (~80%).
-- ============================================================================
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /tmp/create_impulse_fade_config.sql

-- ============================================================================
-- 1. GLOBAL CONFIG TABLE (column-per-parameter, single active row)
-- ============================================================================

CREATE TABLE IF NOT EXISTS impulse_fade_global_config (
    id SERIAL PRIMARY KEY,

    strategy_name VARCHAR(50) NOT NULL DEFAULT 'IMPULSE_FADE',
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Session window (UTC hours, inclusive on both ends)
    session_start_hour INTEGER NOT NULL DEFAULT 18,
    session_end_hour INTEGER NOT NULL DEFAULT 22,

    -- ATR body threshold
    atr_body_multiplier FLOAT NOT NULL DEFAULT 2.2,
    atr_period INTEGER NOT NULL DEFAULT 14,
    -- Hard ceiling on ATR (pips) to block post-news spikes
    max_atr_pips FLOAT NOT NULL DEFAULT 15.0,

    -- Risk parameters
    fixed_stop_loss_pips FLOAT NOT NULL DEFAULT 12.0,
    fixed_take_profit_pips FLOAT NOT NULL DEFAULT 8.0,

    -- Time-based exit (candles at 5m cadence; 36 = 3 hours)
    time_stop_candles INTEGER NOT NULL DEFAULT 36,

    -- Confidence
    min_confidence FLOAT NOT NULL DEFAULT 0.50,
    max_confidence FLOAT NOT NULL DEFAULT 0.85,

    -- Cooldown between signals per pair
    signal_cooldown_minutes INTEGER NOT NULL DEFAULT 60,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO impulse_fade_global_config DEFAULT VALUES
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 2. PER-PAIR OVERRIDES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS impulse_fade_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(60) NOT NULL UNIQUE,
    pair_name VARCHAR(10),

    -- Enablement — all pairs start disabled/monitor-only
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    is_traded BOOLEAN NOT NULL DEFAULT FALSE,
    monitor_only BOOLEAN NOT NULL DEFAULT TRUE,

    -- Per-pair scalar overrides (NULL = use global)
    atr_body_multiplier FLOAT,
    max_atr_pips FLOAT,
    fixed_stop_loss_pips FLOAT,
    fixed_take_profit_pips FLOAT,
    min_confidence FLOAT,
    max_confidence FLOAT,
    signal_cooldown_minutes INTEGER,

    -- JSONB ablation surface
    parameter_overrides JSONB NOT NULL DEFAULT '{}',

    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- USDCAD: primary research pair, enabled for monitor-only observation
-- All others disabled until USDCAD forward-test passes (PF≥1.5, n≥30)
INSERT INTO impulse_fade_pair_overrides (epic, pair_name, is_enabled, is_traded, monitor_only)
VALUES
    ('CS.D.USDCAD.MINI.IP', 'USDCAD', TRUE,  FALSE, TRUE),
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', FALSE, FALSE, TRUE),
    ('CS.D.GBPUSD.MINI.IP', 'GBPUSD', FALSE, FALSE, TRUE),
    ('CS.D.USDJPY.MINI.IP', 'USDJPY', FALSE, FALSE, TRUE),
    ('CS.D.AUDUSD.MINI.IP', 'AUDUSD', FALSE, FALSE, TRUE),
    ('CS.D.USDCHF.MINI.IP', 'USDCHF', FALSE, FALSE, TRUE),
    ('CS.D.NZDUSD.MINI.IP', 'NZDUSD', FALSE, FALSE, TRUE),
    ('CS.D.EURJPY.MINI.IP', 'EURJPY', FALSE, FALSE, TRUE),
    ('CS.D.AUDJPY.MINI.IP', 'AUDJPY', FALSE, FALSE, TRUE)
ON CONFLICT (epic) DO NOTHING;

-- ============================================================================
-- 3. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_impulse_fade_global_active
    ON impulse_fade_global_config (is_active);

CREATE INDEX IF NOT EXISTS idx_impulse_fade_pair_enabled
    ON impulse_fade_pair_overrides (is_enabled);

-- ============================================================================
-- 4. VERIFICATION
-- ============================================================================

SELECT 'IMPULSE_FADE tables created' AS status;

SELECT id, version, session_start_hour, session_end_hour,
       atr_body_multiplier, atr_period, max_atr_pips,
       fixed_stop_loss_pips, fixed_take_profit_pips, signal_cooldown_minutes
FROM impulse_fade_global_config
WHERE is_active = TRUE;

SELECT epic, pair_name, is_enabled, monitor_only
FROM impulse_fade_pair_overrides
ORDER BY pair_name;
