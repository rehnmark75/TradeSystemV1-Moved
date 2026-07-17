-- ============================================================================
-- Auto-Backtest Layer — promotion-candidate backtest runs (audit + notify feed)
-- Database: strategy_config
-- ============================================================================
-- The auto-backtest container scans monitor_only_outcomes for cells whose
-- bracket-independent edge metrics clear the trigger gates, runs a
-- bt.py --live-parity backtest for each, records the verdict here.
-- Consumers:
--   * system-monitor polls WHERE notified_at IS NULL AND status is terminal,
--     pushes Telegram, stamps notified_at (at-least-once, survives restarts).
--   * (optional, later) trading-ui page over the run history.
--
-- Apply:
--   docker exec -i postgres psql -U postgres -d strategy_config \
--     < worker/app/forex_scanner/migrations/create_auto_backtest_runs.sql

CREATE TABLE IF NOT EXISTS auto_backtest_runs (
    id              SERIAL PRIMARY KEY,
    strategy        VARCHAR(64)  NOT NULL,
    epic            VARCHAR(64)  NOT NULL,
    pair            VARCHAR(32),
    environment     VARCHAR(16)  NOT NULL DEFAULT 'demo',
    trigger_metrics JSONB,        -- {n, edge_ratio, pct_mfe_favorable, dead_on_arrival_pct,
                                  --  avg_mfe, avg_mae, ref_pf, per_month, window_days}
    backtest_days   INT          NOT NULL,
    command         TEXT,         -- exact bt.py command executed
    status          VARCHAR(16)  NOT NULL DEFAULT 'PENDING' CHECK (status IN
                        ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    results         JSONB,        -- {pf, win_rate_pct, expectancy_pips, winners, losers,
                                  --  breakevens, total_closed, parse_ok}
    verdict         VARCHAR(24)  CHECK (verdict IN
                        ('PROMOTION_CANDIDATE',  -- passed the bt gate: human review next
                         'MARGINAL',             -- PF ~1.0-1.3: not compelling
                         'NO_GO',                -- PF < 1.0 or too few trades
                         'NO_SIGNALS',           -- backtest produced no closed trades
                         'UNKNOWN')),            -- output could not be parsed
    error           TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    notified_at     TIMESTAMPTZ   -- stamped by system-monitor after Telegram send
);

CREATE INDEX IF NOT EXISTS idx_auto_backtest_runs_unnotified
  ON auto_backtest_runs (created_at)
  WHERE notified_at IS NULL AND status IN ('COMPLETED', 'FAILED');

CREATE INDEX IF NOT EXISTS idx_auto_backtest_runs_cell
  ON auto_backtest_runs (strategy, epic, environment, created_at DESC);
