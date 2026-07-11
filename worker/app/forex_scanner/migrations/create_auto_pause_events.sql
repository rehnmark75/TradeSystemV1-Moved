-- ============================================================================
-- Auto-Pause Layer — decision event log (audit + notification feed)
-- Database: strategy_config
-- ============================================================================
-- Every auto-pause decision (including dry-run trips and no-op errors) is
-- recorded here. Two consumers:
--   * system-monitor polls WHERE notified_at IS NULL and pushes Telegram,
--     then stamps notified_at (at-least-once delivery, survives restarts).
--   * trading-ui decay-monitor page shows the recent event feed.
--
-- Apply:
--   docker exec -i postgres psql -U postgres -d strategy_config \
--     < worker/app/forex_scanner/migrations/create_auto_pause_events.sql

CREATE TABLE IF NOT EXISTS auto_pause_events (
    id          SERIAL PRIMARY KEY,
    event_type  VARCHAR(24)  NOT NULL CHECK (event_type IN (
                    'trip',            -- rule fired (always logged, even when enforced)
                    'pause',           -- monitor_only actually flipped on
                    'dry_run_trip',    -- rule fired but dry-run: NOT flipped
                    'resume_proposed', -- resume rule met, propose-only
                    'resumed',         -- fully auto-resumed
                    'flip_noop_error'  -- flip matched 0 rows: cell NOT protected
                )),
    strategy    VARCHAR(64)  NOT NULL,
    epic        VARCHAR(64)  NOT NULL,
    config_set  VARCHAR(16)  NOT NULL DEFAULT 'demo',
    reason      TEXT,
    metrics     JSONB,                 -- {pf, win_rate, n, consecutive_losses, baseline_*...}
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    notified_at TIMESTAMPTZ            -- stamped by system-monitor after Telegram send
);

CREATE INDEX IF NOT EXISTS idx_auto_pause_events_unnotified
  ON auto_pause_events (created_at) WHERE notified_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_auto_pause_events_cell
  ON auto_pause_events (strategy, epic, config_set, created_at DESC);
