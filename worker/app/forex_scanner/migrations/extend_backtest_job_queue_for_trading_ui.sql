-- Migration: Extend backtest_job_queue for trading-ui workflow
-- Purpose: Add fields required by the trading-ui backtests page so it can
-- queue historical-intelligence runs and parameter variations without relying
-- on Streamlit-specific in-memory state.

ALTER TABLE backtest_job_queue
  ADD COLUMN IF NOT EXISTS use_historical_intelligence BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS variation_config JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS progress JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS recent_output JSONB DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS cancel_requested_at TIMESTAMP DEFAULT NULL;

COMMENT ON COLUMN backtest_job_queue.use_historical_intelligence
  IS 'When true, replay historical market intelligence during backtest execution.';

COMMENT ON COLUMN backtest_job_queue.variation_config
  IS 'Optional parameter variation payload: {enabled, param_grid, workers, rank_by, top_n}.';

COMMENT ON COLUMN backtest_job_queue.progress
  IS 'Live job progress payload: {phase, elapsed_seconds, last_activity, current, total}.';

COMMENT ON COLUMN backtest_job_queue.recent_output
  IS 'Rolling tail of recent stdout lines from the active or completed backtest job.';

COMMENT ON COLUMN backtest_job_queue.cancel_requested_at
  IS 'Set when a queued or running job is asked to cancel from trading-ui.';
