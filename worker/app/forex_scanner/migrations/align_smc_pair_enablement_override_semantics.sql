-- Align SMC Simple pair enablement with long-term override semantics.
--
-- New meaning of smc_simple_pair_overrides.is_enabled:
--   NULL  = inherit smc_simple_global_config.enabled_pairs
--   TRUE  = force enable this epic
--   FALSE = force disable this epic

ALTER TABLE smc_simple_pair_overrides
    ALTER COLUMN is_enabled DROP DEFAULT;

COMMENT ON COLUMN smc_simple_pair_overrides.is_enabled IS
    'Tri-state pair enablement override. NULL=inherits global enabled_pairs, TRUE=force enabled, FALSE=force disabled.';
