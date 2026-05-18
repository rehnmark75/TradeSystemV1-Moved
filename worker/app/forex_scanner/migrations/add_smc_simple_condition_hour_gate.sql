-- Add condition-aware blocks for SMC_SIMPLE scalp hour buckets.
--
-- These columns keep the hour bucket as context, but block only the market
-- conditions that have shown weak realized performance.

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_enabled BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_mode VARCHAR(20) DEFAULT 'ACTIVE',
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_target_epics TEXT DEFAULT 'CS.D.EURUSD.CEEM.IP',
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_compression_buckets TEXT DEFAULT 'BULL:h00_03,BULL:h08_11',
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_compression_atr_max NUMERIC(8, 3) DEFAULT 50.0,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_compression_bb_max NUMERIC(8, 3) DEFAULT 50.0,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_extreme_buckets TEXT DEFAULT 'BEAR:h08_11',
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_extreme_regime VARCHAR(40) DEFAULT 'high_volatility',
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_extreme_volatility_state VARCHAR(40) DEFAULT 'extreme',
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_dynamic_enabled BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_static_fallback_enabled BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_dynamic_min_trades INTEGER DEFAULT 8,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_dynamic_block_profit_factor NUMERIC(8, 3) DEFAULT 0.75,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_dynamic_block_expectancy_pips NUMERIC(8, 3) DEFAULT -0.25,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_expanded_percentile_min NUMERIC(8, 3) DEFAULT 70.0,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_low_quality_max NUMERIC(8, 3) DEFAULT 0.50,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_low_momentum_max NUMERIC(8, 3) DEFAULT 0.50,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_low_confluence_max NUMERIC(8, 3) DEFAULT 0.35,
    ADD COLUMN IF NOT EXISTS scalp_condition_hour_gate_low_efficiency_max NUMERIC(8, 3) DEFAULT 0.20;

COMMENT ON COLUMN smc_simple_global_config.scalp_condition_hour_gate_enabled IS
    'Enable condition-aware blocks inside dynamic scalp hour buckets.';

COMMENT ON COLUMN smc_simple_global_config.scalp_condition_hour_gate_compression_buckets IS
    'Comma-separated direction_hour4 buckets blocked when ATR and BB width percentiles are below configured maxima.';

COMMENT ON COLUMN smc_simple_global_config.scalp_condition_hour_gate_extreme_buckets IS
    'Comma-separated direction_hour4 buckets blocked in the configured extreme regime/volatility state.';

COMMENT ON COLUMN smc_simple_global_config.scalp_condition_hour_gate_dynamic_enabled IS
    'Enable learned condition labels inside direction/hour buckets using recent closed trades or seed backtest rows.';

COMMENT ON COLUMN smc_simple_global_config.scalp_condition_hour_gate_static_fallback_enabled IS
    'Keep configured EURUSD condition blocks active when learned condition labels do not yet have enough evidence.';
