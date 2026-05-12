-- Add opt-in adaptive bucket gate settings for SMC_SIMPLE.
-- Defaults are disabled / monitoring so this migration does not change live behavior.

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_enabled BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_mode VARCHAR(20) DEFAULT 'MONITORING',
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_bucket_mode VARCHAR(40) DEFAULT 'direction',
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_window INTEGER DEFAULT 12,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_min_trades INTEGER DEFAULT 8,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_min_win_rate NUMERIC DEFAULT 0.45,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_min_expectancy_pips NUMERIC DEFAULT -0.25,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_min_pip_samples INTEGER DEFAULT 4,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_pause_hours NUMERIC DEFAULT 48.0,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_probe_after_hours NUMERIC DEFAULT 12.0,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_exploration_rate NUMERIC DEFAULT 0.02,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_max_confidence NUMERIC DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_cache_ttl_seconds INTEGER DEFAULT 300;

COMMENT ON COLUMN smc_simple_global_config.adaptive_bucket_gate_enabled IS
    'Enable live SMC_SIMPLE adaptive bucket gate. Disabled by default.';
COMMENT ON COLUMN smc_simple_global_config.adaptive_bucket_gate_mode IS
    'MONITORING annotates/logs only; ACTIVE blocks weak paused buckets.';
COMMENT ON COLUMN smc_simple_global_config.adaptive_bucket_gate_bucket_mode IS
    'Coarse learning bucket: direction, direction_session, or direction_regime.';
COMMENT ON COLUMN smc_simple_global_config.adaptive_bucket_gate_exploration_rate IS
    'Deterministic probe rate for paused buckets after probe_after_hours.';
COMMENT ON COLUMN smc_simple_global_config.adaptive_bucket_gate_max_confidence IS
    'Optional confidence cap applied by adaptive gate. NULL disables the cap.';

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS direction_quality_gate_enabled BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS direction_quality_gate_mode VARCHAR(20) DEFAULT 'MONITORING',
    ADD COLUMN IF NOT EXISTS direction_quality_gate_target_epics TEXT DEFAULT 'CS.D.EURUSD.CEEM.IP',
    ADD COLUMN IF NOT EXISTS direction_quality_bull_block_start_hour INTEGER DEFAULT 15,
    ADD COLUMN IF NOT EXISTS direction_quality_bull_block_end_hour INTEGER DEFAULT 18,
    ADD COLUMN IF NOT EXISTS direction_quality_bull_min_confidence NUMERIC DEFAULT 0.60,
    ADD COLUMN IF NOT EXISTS direction_quality_bull_rsi_min NUMERIC DEFAULT 50.0,
    ADD COLUMN IF NOT EXISTS direction_quality_bull_rsi_max NUMERIC DEFAULT 70.0,
    ADD COLUMN IF NOT EXISTS direction_quality_bull_require_ema21_gt_ema50 BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS direction_quality_bull_macd_mode VARCHAR(20) DEFAULT 'pullback',
    ADD COLUMN IF NOT EXISTS direction_quality_bear_block_start_hour INTEGER DEFAULT 14,
    ADD COLUMN IF NOT EXISTS direction_quality_bear_block_end_hour INTEGER DEFAULT 17,
    ADD COLUMN IF NOT EXISTS direction_quality_bear_macd_mode VARCHAR(30) DEFAULT 'aligned_or_evening',
    ADD COLUMN IF NOT EXISTS direction_quality_bear_evening_start_hour INTEGER DEFAULT 19;

COMMENT ON COLUMN smc_simple_global_config.direction_quality_gate_enabled IS
    'Enable asymmetric direction quality gate. Runs in backtest and live.';
COMMENT ON COLUMN smc_simple_global_config.direction_quality_bull_macd_mode IS
    'BULL MACD behavior: pullback, aligned, or off.';
COMMENT ON COLUMN smc_simple_global_config.direction_quality_bear_macd_mode IS
    'BEAR MACD behavior: aligned, aligned_or_evening, or off.';
