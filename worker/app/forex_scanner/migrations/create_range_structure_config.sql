-- ============================================================================
-- RANGE_STRUCTURE Strategy Database Configuration v1.0.0
-- ============================================================================
-- Non-trending market-structure strategy: liquidity sweep + rejection wick at
-- range extremes (EQH/EQL). Ships monitor-only on USDJPY + JPY crosses +
-- USDCHF/USDCAD; other pairs disabled until the basket proves PF >= 1.30 at
-- n >= 120 per the Apr 20 2026 plan.
--
-- Schema style: direct columns (NOT the parameter_name/value template). All
-- global knobs are typed; pair overrides are nullable columns that fall back
-- to global values when NULL.
--
-- Apply:
--   docker exec postgres psql -U postgres -d strategy_config \
--     -f /app/forex_scanner/migrations/create_range_structure_config.sql
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. GLOBAL CONFIG (single active row)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS range_structure_global_config (
    id                             SERIAL PRIMARY KEY,
    is_active                      BOOLEAN       NOT NULL DEFAULT TRUE,

    -- Range build / sweep detection
    range_lookback_bars            INTEGER       NOT NULL DEFAULT 40,
    sweep_penetration_pips         NUMERIC(5,2)  NOT NULL DEFAULT 1.5,
    rejection_wick_ratio           NUMERIC(3,2)  NOT NULL DEFAULT 0.60,

    -- Confluence / targets
    ob_fvg_confluence_required     BOOLEAN       NOT NULL DEFAULT TRUE,
    equilibrium_target_enabled     BOOLEAN       NOT NULL DEFAULT TRUE,

    -- R:R floor (RANGING_MARKET v4.0 died from inverse R:R — non-negotiable)
    min_rr_ratio                   NUMERIC(3,2)  NOT NULL DEFAULT 1.33,

    -- Hard ADX gates (ALWAYS enforced, no routing bypass loophole)
    adx_hard_ceiling_primary       NUMERIC(4,1)  NOT NULL DEFAULT 20.0,
    adx_hard_ceiling_htf           NUMERIC(4,1)  NOT NULL DEFAULT 22.0,
    adx_period                     INTEGER       NOT NULL DEFAULT 14,

    -- SL/TP clamps (pips, absolute)
    sl_pips_min                    NUMERIC(4,1)  NOT NULL DEFAULT 6.0,
    sl_pips_max                    NUMERIC(4,1)  NOT NULL DEFAULT 12.0,
    tp_pips_min                    NUMERIC(4,1)  NOT NULL DEFAULT 10.0,
    tp_pips_max                    NUMERIC(4,1)  NOT NULL DEFAULT 18.0,
    sl_buffer_pips                 NUMERIC(4,1)  NOT NULL DEFAULT 1.0,

    -- HTF bias band: |score - 0.5| <= htf_bias_neutral_band
    -- i.e. default 0.40 => accept scores in [0.10, 0.90] (mild-to-neutral).
    htf_bias_neutral_band          NUMERIC(3,2)  NOT NULL DEFAULT 0.40,

    -- Cooldown / confidence
    signal_cooldown_minutes        INTEGER       NOT NULL DEFAULT 60,
    min_confidence                 NUMERIC(3,2)  NOT NULL DEFAULT 0.55,
    max_confidence                 NUMERIC(3,2)  NOT NULL DEFAULT 0.80,

    -- Routing trust (hard ADX gates ALWAYS win; this only affects the soft path)
    trust_regime_routing           BOOLEAN       NOT NULL DEFAULT TRUE,

    -- Timeframes
    primary_timeframe              VARCHAR(8)    NOT NULL DEFAULT '15m',
    confirmation_timeframe         VARCHAR(8)    NOT NULL DEFAULT '1h',

    strategy_version               VARCHAR(16)   NOT NULL DEFAULT '1.0.0',

    notes                          TEXT,
    created_at                     TIMESTAMPTZ   DEFAULT NOW(),
    updated_at                     TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_range_structure_global_active
    ON range_structure_global_config (is_active);

-- Seed the single active row. Guarded so re-running the migration is a no-op.
INSERT INTO range_structure_global_config (is_active, notes)
SELECT TRUE, 'Initial seed — RANGE_STRUCTURE v1.0.0 per plan analyze-the-data-of-clever-wilkes'
WHERE NOT EXISTS (
    SELECT 1 FROM range_structure_global_config WHERE is_active = TRUE
);


-- ----------------------------------------------------------------------------
-- 2. PER-PAIR OVERRIDES
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS range_structure_pair_overrides (
    id                             SERIAL PRIMARY KEY,
    epic                           VARCHAR(50)   UNIQUE NOT NULL,
    pair_name                      VARCHAR(16),

    -- Enable / trading flags
    is_enabled                     BOOLEAN       NOT NULL DEFAULT TRUE,
    is_traded                      BOOLEAN       NOT NULL DEFAULT FALSE,
    monitor_only                   BOOLEAN       NOT NULL DEFAULT TRUE,

    -- Nullable overrides (NULL => use global)
    adx_hard_ceiling_primary       NUMERIC(4,1),
    adx_hard_ceiling_htf           NUMERIC(4,1),
    sweep_penetration_pips         NUMERIC(5,2),
    rejection_wick_ratio           NUMERIC(3,2),
    sl_pips_min                    NUMERIC(4,1),
    sl_pips_max                    NUMERIC(4,1),
    tp_pips_min                    NUMERIC(4,1),
    tp_pips_max                    NUMERIC(4,1),
    min_rr_ratio                   NUMERIC(3,2),
    min_confidence                 NUMERIC(3,2),
    max_confidence                 NUMERIC(3,2),
    signal_cooldown_minutes        INTEGER,
    range_lookback_bars            INTEGER,
    ob_fvg_confluence_required     BOOLEAN,

    -- Free-form JSONB bag for arbitrary experimental overrides
    parameter_overrides            JSONB         DEFAULT '{}'::jsonb,

    notes                          TEXT,
    disabled_reason                TEXT,
    created_at                     TIMESTAMPTZ   DEFAULT NOW(),
    updated_at                     TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_range_structure_pair_enabled
    ON range_structure_pair_overrides (is_enabled);
CREATE INDEX IF NOT EXISTS idx_range_structure_pair_epic
    ON range_structure_pair_overrides (epic);


-- ----------------------------------------------------------------------------
-- 3. SEED PAIR ROWS — v1.0 pilot basket + explicit disables
-- ----------------------------------------------------------------------------

-- USDJPY: primary target. Tight ADX gates (per-pair override) + 8-pip SL floor
-- because 15m ranging ATR median is ~5.5 pips (7-pip SL would be ~1.3 ATR).
INSERT INTO range_structure_pair_overrides
    (epic, pair_name, is_enabled, is_traded, monitor_only,
     adx_hard_ceiling_primary, adx_hard_ceiling_htf, sl_pips_min, notes)
VALUES
    ('CS.D.USDJPY.MINI.IP', 'USDJPY', TRUE, FALSE, TRUE,
     18.0, 20.0, 8.0,
     'Primary target pair; tight ranging ATR (median 15m ATR ~5.5p).')
ON CONFLICT (epic) DO NOTHING;

-- EURJPY / AUDJPY: JPY crosses, monitor-only with defaults. EURJPY is where
-- the EQH/EQL wick-rejection edge (PF 3.85 @ n=11) was observed.
INSERT INTO range_structure_pair_overrides
    (epic, pair_name, is_enabled, is_traded, monitor_only, notes)
VALUES
    ('CS.D.EURJPY.MINI.IP', 'EURJPY', TRUE, FALSE, TRUE,
     'JPY cross; EQH/EQL wick-rejection basis pair.'),
    ('CS.D.AUDJPY.MINI.IP', 'AUDJPY', TRUE, FALSE, TRUE,
     'JPY cross; monitor-only basket member.'),
    ('CS.D.USDCHF.MINI.IP', 'USDCHF', TRUE, FALSE, TRUE,
     'Low-vol quote currency; monitor-only basket member.'),
    ('CS.D.USDCAD.MINI.IP', 'USDCAD', TRUE, FALSE, TRUE,
     'Ranging-friendly FX major; monitor-only basket member.')
ON CONFLICT (epic) DO NOTHING;

-- Explicitly disabled pairs — not part of v1.0 pilot basket. Kept as rows so
-- the UI displays them with a reason rather than silently missing.
INSERT INTO range_structure_pair_overrides
    (epic, pair_name, is_enabled, is_traded, monitor_only, notes, disabled_reason)
VALUES
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', FALSE, FALSE, TRUE,
     'Not in v1.0 pilot basket.',
     'SMC_SIMPLE already covers EURUSD strongly (84% WR); add after v1.0 validation.'),
    ('CS.D.GBPUSD.MINI.IP', 'GBPUSD', FALSE, FALSE, TRUE,
     'Not in v1.0 pilot basket.',
     'Excluded pending larger-n JPY-first validation.'),
    ('CS.D.AUDUSD.MINI.IP', 'AUDUSD', FALSE, FALSE, TRUE,
     'Not in v1.0 pilot basket.',
     'Excluded pending larger-n JPY-first validation.'),
    ('CS.D.NZDUSD.MINI.IP', 'NZDUSD', FALSE, FALSE, TRUE,
     'Not in v1.0 pilot basket.',
     'Excluded pending larger-n JPY-first validation.'),
    ('CS.D.GBPJPY.MINI.IP', 'GBPJPY', FALSE, FALSE, TRUE,
     'Not in v1.0 pilot basket.',
     'High-vol JPY cross; add only after EUR/AUD/USD-JPY show PF >= 1.3.')
ON CONFLICT (epic) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 4. AUDIT TABLE
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS range_structure_config_audit (
    id              SERIAL PRIMARY KEY,
    table_name      VARCHAR(50) NOT NULL,
    record_id       INTEGER     NOT NULL,
    column_name     VARCHAR(100),
    old_value       TEXT,
    new_value       TEXT,
    change_type     VARCHAR(20) NOT NULL,   -- INSERT|UPDATE|DELETE
    changed_by      VARCHAR(100) DEFAULT 'system',
    change_reason   TEXT,
    changed_at      TIMESTAMPTZ DEFAULT NOW()
);


SELECT 'RANGE_STRUCTURE configuration tables created' AS status;
