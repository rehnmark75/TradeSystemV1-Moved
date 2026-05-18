-- Add active dynamic scalp hour gate settings for SMC_SIMPLE.
-- The gate overlays static per-pair scalp_blocked_hours_utc with recent
-- realized performance by epic + UTC hour.

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_enabled BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_mode VARCHAR(20) DEFAULT 'ACTIVE',
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_window INTEGER DEFAULT 24,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_min_trades INTEGER DEFAULT 8,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_lookback_days INTEGER DEFAULT 45,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_block_profit_factor NUMERIC DEFAULT 0.75,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_block_expectancy_pips NUMERIC DEFAULT -0.15,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_release_profit_factor NUMERIC DEFAULT 1.05,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_release_expectancy_pips NUMERIC DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_probe_rate NUMERIC DEFAULT 0.10,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_cache_ttl_seconds INTEGER DEFAULT 300,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_seed_execution_id INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_seed_strategy_name VARCHAR(50) DEFAULT 'SMC_SIMPLE';

COMMENT ON COLUMN smc_simple_global_config.scalp_hour_gate_enabled IS
    'Enable dynamic scalp hour gate over static per-pair blocked hours.';
COMMENT ON COLUMN smc_simple_global_config.scalp_hour_gate_mode IS
    'MONITORING annotates/logs only; ACTIVE blocks dynamic weak hours and static unreleased hours.';
COMMENT ON COLUMN smc_simple_global_config.scalp_hour_gate_probe_rate IS
    'Daily deterministic probe rate for static-blocked hours with insufficient recent evidence.';
COMMENT ON COLUMN smc_simple_global_config.scalp_hour_gate_seed_execution_id IS
    'Optional backtest execution_id used to seed/backtest hour performance without lookahead.';

UPDATE smc_simple_global_config
SET
    scalp_hour_gate_enabled = TRUE,
    scalp_hour_gate_mode = 'ACTIVE'
WHERE is_active = TRUE;

-- The dynamic SMC_SIMPLE scalp hour gate is the single owner of hour gating.
-- Disable the generic LPF bad_hours rule for every active SMC_SIMPLE epic so
-- LPF does not re-block hours that the dynamic gate is probing/releasing.
WITH active_smc_epics AS (
    SELECT DISTINCT g.config_set, p.epic
    FROM smc_simple_global_config g
    JOIN smc_simple_pair_overrides p ON p.config_id = g.id
    WHERE g.is_active = TRUE
)
UPDATE loss_prevention_pair_config lpc
SET
    disabled_rules = (
        SELECT array_agg(DISTINCT rule_name ORDER BY rule_name)
        FROM unnest(coalesce(lpc.disabled_rules, ARRAY[]::text[]) || ARRAY['bad_hours']::text[]) AS rules(rule_name)
    ),
    notes = concat_ws(
        E'\n',
        nullif(lpc.notes, ''),
        'Dynamic SMC_SIMPLE scalp_hour_gate owns hour gating; LPF bad_hours disabled for this epic.'
    ),
    updated_at = now()
FROM active_smc_epics smc
WHERE lpc.config_set = smc.config_set
  AND lpc.epic = smc.epic;

WITH active_smc_epics AS (
    SELECT DISTINCT g.config_set, p.epic
    FROM smc_simple_global_config g
    JOIN smc_simple_pair_overrides p ON p.config_id = g.id
    WHERE g.is_active = TRUE
)
INSERT INTO loss_prevention_pair_config (
    config_set,
    epic,
    disabled_rules,
    notes
)
SELECT
    smc.config_set,
    smc.epic,
    ARRAY['bad_hours']::text[],
    'Dynamic SMC_SIMPLE scalp_hour_gate owns hour gating; LPF bad_hours disabled for this epic.'
FROM active_smc_epics smc
WHERE NOT EXISTS (
    SELECT 1
    FROM loss_prevention_pair_config lpc
    WHERE lpc.config_set = smc.config_set
      AND lpc.epic = smc.epic
);
