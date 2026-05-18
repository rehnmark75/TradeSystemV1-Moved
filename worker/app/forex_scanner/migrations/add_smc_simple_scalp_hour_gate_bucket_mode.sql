-- Add bucket mode support for the dynamic SMC_SIMPLE scalp hour gate.
--
-- hour            = existing behavior, epic + UTC hour
-- hour4           = epic + 4-hour UTC block
-- direction       = epic + signal direction
-- direction_hour  = epic + signal direction + UTC hour
-- direction_hour4 = epic + signal direction + 4-hour UTC block

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS scalp_hour_gate_bucket_mode VARCHAR(40) DEFAULT 'hour';

COMMENT ON COLUMN smc_simple_global_config.scalp_hour_gate_bucket_mode IS
    'Bucket mode for dynamic scalp hour gate: hour, hour4, direction, direction_hour, direction_hour4.';
