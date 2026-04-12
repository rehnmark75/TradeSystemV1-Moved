-- Migration: trailing_pair_config and trailing_config_audit
-- Database: strategy_config
-- Purpose: Move trailing stop configs from hardcoded Python dicts in
--          dev-app/config.py to a DB table with per-environment scoping
--          (config_set = 'demo' | 'live').
--
-- Consumers of this table: dev-app/services/trailing_config_service.py
-- Accessor:                dev-app/config.py :: get_trailing_config_for_epic()

BEGIN;

CREATE TABLE IF NOT EXISTS trailing_pair_config (
    id SERIAL PRIMARY KEY,
    config_set TEXT NOT NULL CHECK (config_set IN ('demo','live')),
    epic TEXT NOT NULL,                         -- 'DEFAULT' for fallback row
    is_scalp BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Stage triggers and locks
    early_breakeven_trigger_points INTEGER,
    early_breakeven_buffer_points INTEGER,
    stage1_trigger_points INTEGER,
    stage1_lock_points INTEGER,
    stage2_trigger_points INTEGER,
    stage2_lock_points INTEGER,
    stage3_trigger_points INTEGER,
    stage3_atr_multiplier NUMERIC(5,2),
    stage3_min_distance INTEGER,
    min_trail_distance INTEGER,
    break_even_trigger_points INTEGER,

    -- Partial close
    enable_partial_close BOOLEAN,
    partial_close_trigger_points INTEGER,
    partial_close_size NUMERIC(4,2),

    -- Metadata
    updated_by TEXT,
    change_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT trailing_unique_scope UNIQUE (config_set, epic, is_scalp)
);

CREATE INDEX IF NOT EXISTS idx_trailing_lookup
    ON trailing_pair_config (config_set, epic, is_scalp)
    WHERE is_active = TRUE;

-- Auto-update updated_at trigger (reuse existing helper if present)
CREATE OR REPLACE FUNCTION trg_trailing_pair_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trailing_pair_config_updated_at ON trailing_pair_config;
CREATE TRIGGER trailing_pair_config_updated_at
    BEFORE UPDATE ON trailing_pair_config
    FOR EACH ROW
    EXECUTE FUNCTION trg_trailing_pair_config_updated_at();

-- Audit trail
CREATE TABLE IF NOT EXISTS trailing_config_audit (
    id SERIAL PRIMARY KEY,
    config_id INTEGER REFERENCES trailing_pair_config(id) ON DELETE SET NULL,
    change_type TEXT NOT NULL,                  -- 'UPDATE' | 'CREATE' | 'DELETE'
    changed_by TEXT,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    change_reason TEXT,
    previous_values JSONB,
    new_values JSONB
);

CREATE INDEX IF NOT EXISTS idx_trailing_audit_config_id
    ON trailing_config_audit (config_id);
CREATE INDEX IF NOT EXISTS idx_trailing_audit_changed_at
    ON trailing_config_audit (changed_at DESC);

COMMIT;
