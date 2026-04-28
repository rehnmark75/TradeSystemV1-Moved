-- Adds post-Claude execution verdict columns to alert_history.
-- Apr 28 2026: ends the recurring "why didn't this approved alert trade?" investigation.

ALTER TABLE alert_history
    ADD COLUMN IF NOT EXISTS block_reason TEXT,
    ADD COLUMN IF NOT EXISTS executed_at  TIMESTAMP WITH TIME ZONE;

CREATE INDEX IF NOT EXISTS ix_alert_history_block_reason
    ON alert_history (block_reason)
    WHERE block_reason IS NOT NULL;

COMMENT ON COLUMN alert_history.block_reason IS
    'Reason a Claude-approved alert did not produce a trade. NULL when executed_at IS NOT NULL. '
    'Values: monitor_only_strategy, monitor_only_pair, cooldown, validation:<reason>, paper_mode_disabled, '
    'executor_error:<msg>, executor_method_unavailable, exception:<msg>.';
COMMENT ON COLUMN alert_history.executed_at IS
    'Timestamp the order was successfully executed via OrderExecutor. NULL when block_reason IS NOT NULL.';
