-- Add early-failure stop controls to database-driven trailing config.
-- These fields are consumed by the XAU backtest trailing engine and are
-- stored here so the tested runner profile can be managed with the rest of
-- trailing_pair_config.

ALTER TABLE trailing_pair_config
    ADD COLUMN IF NOT EXISTS early_failure_stop_enabled BOOLEAN,
    ADD COLUMN IF NOT EXISTS early_failure_check_bars INTEGER,
    ADD COLUMN IF NOT EXISTS early_failure_min_mfe_pips INTEGER,
    ADD COLUMN IF NOT EXISTS early_failure_stop_pips INTEGER;
