-- ============================================================================
-- monitor_only_outcomes: add was_executed (shadow-series completion)
-- Database: forex
-- ============================================================================
-- The analyzer originally evaluated ONLY logged-but-not-executed signals
-- (t.alert_id IS NULL), so executed signals never got a shadow row. That means
-- for actively-traded cells the shadow series excludes exactly the traded
-- signals — unusable as a per-cell decay series (auto-pause Trip Rule B needs
-- the COMPLETE signal population per (strategy, epic)).
--
-- From this migration on, monitor_outcome_analyzer evaluates ALL alert_history
-- signals and stamps was_executed. Consumers that want the original
-- "monitor-only" semantics must filter WHERE was_executed = FALSE.
--
-- Apply:
--   docker exec -i postgres psql -U postgres -d forex \
--     < worker/app/forex_scanner/migrations/add_monitor_only_outcomes_was_executed.sql

ALTER TABLE monitor_only_outcomes
    ADD COLUMN IF NOT EXISTS was_executed BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN monitor_only_outcomes.was_executed IS
  'TRUE when the signal has a trade_log execution. The table holds the ref-grid '
  'outcome for ALL signals; filter was_executed=FALSE for the original '
  'monitor-only population.';

-- Rolling decay-series lookup: last-N resolved outcomes per cell.
CREATE INDEX IF NOT EXISTS idx_monitor_outcomes_decay_lookup
  ON monitor_only_outcomes (strategy, epic, environment, status, signal_timestamp DESC);
