-- Persist adaptive bucket gate annotations in alert_history.
-- Used for MONITORING mode audit before enabling ACTIVE blocking.

ALTER TABLE alert_history
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_state VARCHAR(40),
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_bucket TEXT,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_win_rate NUMERIC,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_trades INTEGER,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_expectancy_pips NUMERIC,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_would_block BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_reason TEXT,
    ADD COLUMN IF NOT EXISTS adaptive_bucket_gate_probe BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_alert_history_adaptive_bucket_gate
    ON alert_history (adaptive_bucket_gate_state, adaptive_bucket_gate_would_block)
    WHERE strategy = 'SMC_SIMPLE';

ALTER TABLE alert_history
    ADD COLUMN IF NOT EXISTS direction_quality_gate_state VARCHAR(40),
    ADD COLUMN IF NOT EXISTS direction_quality_gate_mode VARCHAR(20),
    ADD COLUMN IF NOT EXISTS direction_quality_gate_reason TEXT,
    ADD COLUMN IF NOT EXISTS direction_quality_gate_details JSONB,
    ADD COLUMN IF NOT EXISTS direction_quality_gate_would_block BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_alert_history_direction_quality_gate
    ON alert_history (direction_quality_gate_state, direction_quality_gate_would_block)
    WHERE strategy = 'SMC_SIMPLE';
