-- =============================================================================
-- Trailing Ratio Config (Apr 2026)
-- =============================================================================
-- Moves DEFAULT_TRAILING_RATIOS / DEFAULT_SCALP_TRAILING_RATIOS and per-pair
-- ratio overrides from dev-app/config.py into the strategy_config DB, so that
-- compute_sltp_trailing_config() reads ratios at runtime (with 120s TTL) and
-- trading-ui can edit them without a container rebuild.
--
-- Scope key: (config_set, epic, is_scalp).
--   config_set: 'demo' | 'live'
--   epic:       'DEFAULT' for global baseline, else a full epic string
--   is_scalp:   two independent ratio profiles per env
--
-- Resolution order at runtime:
--   1. Pair-specific row (epic = <epic>, is_scalp = X)
--   2. Baseline row      (epic = 'DEFAULT', is_scalp = X)
--   Columns that are NULL on the pair row inherit from the DEFAULT row.
-- =============================================================================

CREATE TABLE IF NOT EXISTS trailing_ratio_config (
    id                           SERIAL PRIMARY KEY,
    config_set                   TEXT NOT NULL CHECK (config_set IN ('demo', 'live')),
    epic                         TEXT NOT NULL,
    is_scalp                     BOOLEAN NOT NULL DEFAULT FALSE,
    is_active                    BOOLEAN NOT NULL DEFAULT TRUE,

    -- Trigger ratios (× TP_pips)
    early_be_trigger_ratio       NUMERIC(5,3),
    stage1_trigger_ratio         NUMERIC(5,3),
    stage2_trigger_ratio         NUMERIC(5,3),
    stage3_trigger_ratio         NUMERIC(5,3),
    break_even_trigger_ratio     NUMERIC(5,3),
    partial_close_trigger_ratio  NUMERIC(5,3),

    -- Lock ratios (× TP_pips)
    stage1_lock_ratio            NUMERIC(5,3),
    stage2_lock_ratio            NUMERIC(5,3),

    -- Fixed / ratio-based non-trigger values
    early_be_buffer_points       NUMERIC(5,2),
    stage3_atr_multiplier        NUMERIC(5,2),
    stage3_min_distance_ratio    NUMERIC(5,3),
    min_trail_distance_ratio     NUMERIC(5,3),

    -- Minimum floor values (triggers never go below these)
    min_early_be_trigger         INTEGER,
    min_stage1_trigger           INTEGER,
    min_stage1_lock              INTEGER,
    min_stage2_trigger           INTEGER,
    min_stage2_lock              INTEGER,
    min_stage3_trigger           INTEGER,
    min_break_even_trigger       INTEGER,
    min_trail_distance           INTEGER,

    updated_by                   TEXT,
    change_reason                TEXT,
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT trailing_ratio_unique_scope UNIQUE (config_set, epic, is_scalp)
);

CREATE INDEX IF NOT EXISTS idx_trailing_ratio_lookup
    ON trailing_ratio_config (config_set, epic, is_scalp)
    WHERE is_active = TRUE;

-- Keep updated_at fresh on every change
CREATE OR REPLACE FUNCTION trg_trailing_ratio_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trailing_ratio_config_updated_at ON trailing_ratio_config;
CREATE TRIGGER trailing_ratio_config_updated_at
    BEFORE UPDATE ON trailing_ratio_config
    FOR EACH ROW EXECUTE FUNCTION trg_trailing_ratio_config_updated_at();

-- ---------------------------------------------------------------------------
-- Audit table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trailing_ratio_audit (
    id               SERIAL PRIMARY KEY,
    config_id        INTEGER REFERENCES trailing_ratio_config(id) ON DELETE SET NULL,
    change_type      TEXT NOT NULL,       -- 'INSERT' | 'UPDATE' | 'UPSERT' | 'DELETE'
    changed_by       TEXT,
    change_reason    TEXT,
    previous_values  JSONB,
    new_values       JSONB,
    changed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trailing_ratio_audit_config
    ON trailing_ratio_audit (config_id, changed_at DESC);

-- ---------------------------------------------------------------------------
-- Seed rows — mirror the values in dev-app/config.py as of Apr 2026.
-- Ratios for scalp were lowered to 0.33 for early_be / break_even in the
-- same session that produced this migration (prior values were 0.50).
-- ---------------------------------------------------------------------------
INSERT INTO trailing_ratio_config (
    config_set, epic, is_scalp,
    early_be_trigger_ratio, stage1_trigger_ratio, stage2_trigger_ratio,
    stage3_trigger_ratio, break_even_trigger_ratio, partial_close_trigger_ratio,
    stage1_lock_ratio, stage2_lock_ratio,
    early_be_buffer_points, stage3_atr_multiplier,
    stage3_min_distance_ratio, min_trail_distance_ratio,
    min_early_be_trigger, min_stage1_trigger, min_stage1_lock,
    min_stage2_trigger, min_stage2_lock, min_stage3_trigger,
    min_break_even_trigger, min_trail_distance,
    updated_by, change_reason
) VALUES
-- demo, non-scalp (DEFAULT_TRAILING_RATIOS)
('demo', 'DEFAULT', FALSE,
 0.56, 0.72, 1.00, 1.30, 0.56, 0.80,
 0.33, 0.65,
 1.5, 2.0,
 0.33, 0.33,
 5, 8, 3, 12, 8, 15, 5, 5,
 'migration', 'Seed from dev-app/config.py DEFAULT_TRAILING_RATIOS'),

-- demo, scalp (DEFAULT_SCALP_TRAILING_RATIOS, post-Apr 2026 tune)
('demo', 'DEFAULT', TRUE,
 0.33, 0.65, 1.00, 1.30, 0.33, 0.80,
 0.33, 0.65,
 1.0, 1.5,
 0.33, 0.33,
 4, 5, 2, 8, 5, 10, 4, 3,
 'migration', 'Seed from dev-app/config.py DEFAULT_SCALP_TRAILING_RATIOS'),

-- live, non-scalp
('live', 'DEFAULT', FALSE,
 0.56, 0.72, 1.00, 1.30, 0.56, 0.80,
 0.33, 0.65,
 1.5, 2.0,
 0.33, 0.33,
 5, 8, 3, 12, 8, 15, 5, 5,
 'migration', 'Seed from dev-app/config.py DEFAULT_TRAILING_RATIOS'),

-- live, scalp
('live', 'DEFAULT', TRUE,
 0.33, 0.65, 1.00, 1.30, 0.33, 0.80,
 0.33, 0.65,
 1.0, 1.5,
 0.33, 0.33,
 4, 5, 2, 8, 5, 10, 4, 3,
 'migration', 'Seed from dev-app/config.py DEFAULT_SCALP_TRAILING_RATIOS')
ON CONFLICT (config_set, epic, is_scalp) DO NOTHING;
