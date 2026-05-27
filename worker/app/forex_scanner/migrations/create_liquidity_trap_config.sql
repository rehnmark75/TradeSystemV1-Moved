-- ============================================================================
-- LIQUIDITY_TRAP Strategy Database Configuration
-- Fades failed breakouts of prior-day high/low on 1H candle close.
-- Direction is against the trap:
--   bar sweeps above PDH but closes below PDH → SELL (institutional trap)
--   bar sweeps below PDL but closes above PDL → BUY
-- ============================================================================
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /tmp/create_liquidity_trap_config.sql

-- ============================================================================
-- 1. GLOBAL CONFIG TABLE (column-per-parameter, single active row)
-- ============================================================================

CREATE TABLE IF NOT EXISTS liquidity_trap_global_config (
    id SERIAL PRIMARY KEY,

    strategy_name VARCHAR(50) NOT NULL DEFAULT 'LIQUIDITY_TRAP',
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Detection params
    trigger_timeframe VARCHAR(10) NOT NULL DEFAULT '1h',
    daily_lookback_days INTEGER NOT NULL DEFAULT 1,
    breakout_buffer_pips FLOAT NOT NULL DEFAULT 1.0,   -- min pips above/below PDH/PDL the wick must reach
    wick_fraction_min FLOAT NOT NULL DEFAULT 0.35,     -- (wick_tail / total_range) minimum
    require_opposing_body BOOLEAN NOT NULL DEFAULT TRUE, -- close must be on opposite side

    -- ATR guard
    atr_period INTEGER NOT NULL DEFAULT 14,
    min_atr_pips FLOAT NOT NULL DEFAULT 8.0,
    max_atr_pips FLOAT NOT NULL DEFAULT 30.0,

    -- Risk
    fixed_stop_loss_pips FLOAT NOT NULL DEFAULT 14.0,  -- SL above swept wick + buffer
    fixed_take_profit_pips FLOAT NOT NULL DEFAULT 20.0,
    sl_buffer_pips FLOAT NOT NULL DEFAULT 3.0,         -- extra pips above/below bar extreme for SL
    min_sl_pips FLOAT NOT NULL DEFAULT 8.0,
    max_sl_pips FLOAT NOT NULL DEFAULT 22.0,
    min_rr_ratio FLOAT NOT NULL DEFAULT 1.5,
    max_tp_pips FLOAT NOT NULL DEFAULT 35.0,

    -- Session (enabled_hours as JSONB list of UTC hours)
    enabled_hours_default JSONB NOT NULL DEFAULT '[7,8,9,10,11,12,13,14,15,16,17,18,19,20]',

    -- Confidence
    min_confidence FLOAT NOT NULL DEFAULT 0.55,
    max_confidence FLOAT NOT NULL DEFAULT 0.85,

    -- Cooldown
    signal_cooldown_minutes INTEGER NOT NULL DEFAULT 90,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO liquidity_trap_global_config DEFAULT VALUES
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 2. PER-PAIR OVERRIDES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS liquidity_trap_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(60) NOT NULL,
    pair_name VARCHAR(10),

    -- Config set: 3=demo, 2=live
    config_id INTEGER NOT NULL DEFAULT 3,

    -- Enablement — validated pairs start enabled/monitor-only; others disabled
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    is_traded BOOLEAN NOT NULL DEFAULT FALSE,
    monitor_only BOOLEAN NOT NULL DEFAULT TRUE,

    -- Per-pair scalar overrides (NULL = use global)
    fixed_stop_loss_pips FLOAT,
    fixed_take_profit_pips FLOAT,
    min_atr_pips FLOAT,
    max_atr_pips FLOAT,
    min_confidence FLOAT,
    max_confidence FLOAT,
    signal_cooldown_minutes INTEGER,

    -- Per-pair session hours override (NULL = use global enabled_hours_default)
    enabled_hours JSONB,

    -- JSONB ablation surface
    parameter_overrides JSONB NOT NULL DEFAULT '{}',

    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE (epic, config_id)
);

-- Seed demo rows (config_id=3):
-- EURJPY + USDJPY are the validated pairs (enabled, monitor-only)
-- All others disabled until forward-test gate passes (n>=50/WR>=58%/PF>=1.40)
INSERT INTO liquidity_trap_pair_overrides (epic, pair_name, config_id, is_enabled, is_traded, monitor_only)
VALUES
    ('CS.D.EURJPY.MINI.IP', 'EURJPY', 3, TRUE,  FALSE, TRUE),
    ('CS.D.USDJPY.MINI.IP', 'USDJPY', 3, TRUE,  FALSE, TRUE),
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', 3, FALSE, FALSE, TRUE),
    ('CS.D.GBPUSD.MINI.IP', 'GBPUSD', 3, FALSE, FALSE, TRUE),
    ('CS.D.USDCAD.MINI.IP', 'USDCAD', 3, FALSE, FALSE, TRUE),
    ('CS.D.AUDUSD.MINI.IP', 'AUDUSD', 3, FALSE, FALSE, TRUE),
    ('CS.D.USDCHF.MINI.IP', 'USDCHF', 3, FALSE, FALSE, TRUE),
    ('CS.D.NZDUSD.MINI.IP', 'NZDUSD', 3, FALSE, FALSE, TRUE),
    ('CS.D.AUDJPY.MINI.IP', 'AUDJPY', 3, FALSE, FALSE, TRUE)
ON CONFLICT (epic, config_id) DO NOTHING;

-- ============================================================================
-- 3. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_liquidity_trap_global_active
    ON liquidity_trap_global_config (is_active);

CREATE INDEX IF NOT EXISTS idx_liquidity_trap_pair_enabled
    ON liquidity_trap_pair_overrides (is_enabled);

-- ============================================================================
-- 4. VERIFICATION
-- ============================================================================

SELECT 'LIQUIDITY_TRAP tables created' AS status;

SELECT id, version, trigger_timeframe, daily_lookback_days,
       breakout_buffer_pips, wick_fraction_min, require_opposing_body,
       min_atr_pips, max_atr_pips,
       fixed_stop_loss_pips, fixed_take_profit_pips,
       sl_buffer_pips, min_sl_pips, max_sl_pips, min_rr_ratio, max_tp_pips,
       min_confidence, max_confidence, signal_cooldown_minutes
FROM liquidity_trap_global_config
WHERE is_active = TRUE;

SELECT epic, pair_name, config_id, is_enabled, monitor_only
FROM liquidity_trap_pair_overrides
ORDER BY pair_name, config_id;
