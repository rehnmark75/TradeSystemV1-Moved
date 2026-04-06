-- Migration: Per-pair LPF configuration
-- Database: strategy_config
-- Purpose: Allow per-pair overrides for LPF threshold, block mode, rule disable, and penalty overrides

\c strategy_config;

CREATE TABLE IF NOT EXISTS loss_prevention_pair_config (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    is_enabled BOOLEAN DEFAULT NULL,              -- NULL=use global, True/False=override
    block_mode VARCHAR(20) DEFAULT NULL,           -- NULL=use global, 'monitor'/'block'
    penalty_threshold NUMERIC(4,2) DEFAULT NULL,   -- NULL=use global threshold
    disabled_rules TEXT[] DEFAULT NULL,             -- Rule names to skip for this pair
    rule_penalty_overrides JSONB DEFAULT NULL,      -- {"rule_name": 0.35, ...} per-rule penalty overrides
    notes TEXT,                                     -- Human-readable reason for config
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lpf_pair_config_epic
    ON loss_prevention_pair_config(epic);

-- Auto-update trigger (reuses existing function from strategy_config DB)
DROP TRIGGER IF EXISTS update_lpf_pair_config_updated_at ON loss_prevention_pair_config;
CREATE TRIGGER update_lpf_pair_config_updated_at
    BEFORE UPDATE ON loss_prevention_pair_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE loss_prevention_pair_config IS
    'Per-pair LPF overrides. NULL columns fall through to loss_prevention_config global settings.';
