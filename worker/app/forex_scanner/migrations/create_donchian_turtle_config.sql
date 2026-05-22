-- ============================================================================
-- DONCHIAN_TURTLE Strategy Database Configuration
-- Classic Turtle Trading (S1: 20-bar entry / 10-bar exit) applied to 1H FX bars.
-- Long-only initially (Long PF 1.21 vs Short PF 1.04 in 4-year backtest).
-- Hard stop: 2×ATR from entry. Exit: opposite-side 10-bar channel breach.
-- Target pairs: USDJPY, USDCHF, EURJPY (all monitor-only at launch).
-- ============================================================================
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /tmp/create_donchian_turtle_config.sql

-- ============================================================================
-- 1. GLOBAL CONFIG TABLE (column-per-parameter, single active row)
-- ============================================================================

CREATE TABLE IF NOT EXISTS donchian_turtle_global_config (
    id SERIAL PRIMARY KEY,

    strategy_name VARCHAR(50) NOT NULL DEFAULT 'DONCHIAN_TURTLE',
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Channel parameters (S1 system)
    entry_bars INTEGER NOT NULL DEFAULT 20,   -- Donchian entry channel lookback
    exit_bars INTEGER NOT NULL DEFAULT 10,    -- Donchian exit channel lookback
    atr_period INTEGER NOT NULL DEFAULT 14,   -- ATR period for hard stop
    atr_stop_multiplier FLOAT NOT NULL DEFAULT 2.0,  -- hard stop = N × ATR

    -- Direction gate
    long_only BOOLEAN NOT NULL DEFAULT TRUE,  -- disable shorts until 6-month live review

    -- Risk fallbacks (used when ATR unavailable)
    fixed_stop_loss_pips FLOAT NOT NULL DEFAULT 50.0,
    fixed_take_profit_pips FLOAT NOT NULL DEFAULT 200.0,  -- safety cap; trailing exits first

    -- Confidence
    min_confidence FLOAT NOT NULL DEFAULT 0.50,
    max_confidence FLOAT NOT NULL DEFAULT 0.95,

    -- Cooldown (1H bars; one valid breakout per pair per day is typical)
    signal_cooldown_minutes INTEGER NOT NULL DEFAULT 240,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO donchian_turtle_global_config DEFAULT VALUES
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 2. PER-PAIR OVERRIDES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS donchian_turtle_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(60) NOT NULL UNIQUE,
    pair_name VARCHAR(10),

    -- Enablement — target pairs monitor-only at launch
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    is_traded BOOLEAN NOT NULL DEFAULT FALSE,
    monitor_only BOOLEAN NOT NULL DEFAULT TRUE,

    -- Per-pair scalar overrides (NULL = use global)
    entry_bars INTEGER,
    exit_bars INTEGER,
    atr_stop_multiplier FLOAT,
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

-- USDJPY: strongest pair (PF 1.37, +5,563 pips / 4yr)
-- USDCHF: second pair (PF 1.22)
-- EURJPY: third pair (PF 1.18)
-- AUDUSD: excluded — no edge (PF 0.99 / mean-reverts on 1H)
-- All others: disabled
INSERT INTO donchian_turtle_pair_overrides
    (epic, pair_name, is_enabled, is_traded, monitor_only, fixed_stop_loss_pips, notes)
VALUES
    ('CS.D.USDJPY.MINI.IP', 'USDJPY', TRUE,  FALSE, TRUE, 55.0, 'Primary pair: PF 1.37, 4yr backtest'),
    ('CS.D.USDCHF.MINI.IP', 'USDCHF', TRUE,  FALSE, TRUE, 38.0, 'Second pair: PF 1.22'),
    ('CS.D.EURJPY.MINI.IP', 'EURJPY', TRUE,  FALSE, TRUE, 60.0, 'Third pair: PF 1.18'),
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', FALSE, FALSE, TRUE, NULL, 'Marginal edge PF 1.06 — disabled'),
    ('CS.D.GBPUSD.MINI.IP', 'GBPUSD', FALSE, FALSE, TRUE, NULL, 'Marginal edge PF 1.03 — disabled'),
    ('CS.D.AUDUSD.MINI.IP', 'AUDUSD', FALSE, FALSE, TRUE, NULL, 'No edge PF 0.99 — excluded'),
    ('CS.D.USDCAD.MINI.IP', 'USDCAD', FALSE, FALSE, TRUE, NULL, 'Marginal edge PF 1.02 — disabled'),
    ('CS.D.NZDUSD.MINI.IP', 'NZDUSD', FALSE, FALSE, TRUE, NULL, 'Not tested — disabled'),
    ('CS.D.AUDJPY.MINI.IP', 'AUDJPY', FALSE, FALSE, TRUE, NULL, 'Not tested — disabled')
ON CONFLICT (epic) DO NOTHING;

-- ============================================================================
-- 3. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_donchian_turtle_global_active
    ON donchian_turtle_global_config (is_active);

CREATE INDEX IF NOT EXISTS idx_donchian_turtle_pair_enabled
    ON donchian_turtle_pair_overrides (is_enabled);

-- ============================================================================
-- 4. VERIFICATION
-- ============================================================================

SELECT 'DONCHIAN_TURTLE tables created' AS status;

SELECT id, version, entry_bars, exit_bars, atr_period, atr_stop_multiplier,
       long_only, fixed_stop_loss_pips, fixed_take_profit_pips, signal_cooldown_minutes
FROM donchian_turtle_global_config
WHERE is_active = TRUE;

SELECT epic, pair_name, is_enabled, monitor_only, fixed_stop_loss_pips
FROM donchian_turtle_pair_overrides
ORDER BY pair_name;
