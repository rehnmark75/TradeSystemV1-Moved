-- Failed followthrough exit tracking on trade_log (forex DB).
-- These columns record outcome data, not config.
-- Config lives in strategy_config.trade_management_guards.

ALTER TABLE trade_log
    ADD COLUMN IF NOT EXISTS failed_followthrough_exit       BOOLEAN   DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS failed_followthrough_exit_at    TIMESTAMP,
    ADD COLUMN IF NOT EXISTS failed_followthrough_mfe_pips   FLOAT,
    ADD COLUMN IF NOT EXISTS failed_followthrough_mae_pips   FLOAT;
