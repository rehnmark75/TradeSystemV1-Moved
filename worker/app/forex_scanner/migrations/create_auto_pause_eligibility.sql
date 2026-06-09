-- ============================================================================
-- Auto-Pause Layer — Eligibility Allowlist (Phase 1)
-- Database: strategy_config
-- ============================================================================
-- A (strategy, epic, config_set) cell receives auto-pause protection ONLY if
-- there is a row here with eligible = TRUE.
--
-- CRITICAL DESIGN RULE — the baseline is FROZEN, never trailing-recent:
--   baseline_pf / baseline_n / baseline_as_of record the *established
--   promotion-time* edge (target: PF > 1.2, n >= 50, > 8 trades/month). They
--   are set by a human when a strategy/pair is promoted to execution and are
--   NOT recomputed from recent performance. Recomputing on a recent window
--   would drop a decaying strategy below the edge bar and strip its protection
--   exactly when it is needed (the whole point of auto-pause is to catch a
--   PROVEN strategy that has started to decay).
--
-- Strategies with no edge baseline must NOT be added with eligible = TRUE:
-- on a marginal/no-edge strategy the trip rule fires ~constantly (≈86% false
-- positives observed on a PF≈1.0 cell), because "decay" is undefined without a
-- baseline. Low-frequency strategies (< 8 trades/month) are also out of scope —
-- the rolling-PF window never fills in time and auto-resume could take months.
-- ============================================================================

CREATE TABLE IF NOT EXISTS auto_pause_eligibility (
    id                 SERIAL PRIMARY KEY,
    strategy           VARCHAR(64)  NOT NULL,          -- matches alert_history.strategy
    epic               VARCHAR(64)  NOT NULL,
    config_set         VARCHAR(16)  NOT NULL DEFAULT 'demo',  -- environment: 'demo' | 'live'
    eligible           BOOLEAN      NOT NULL DEFAULT FALSE,

    -- Frozen, promotion-time edge baseline (documents WHY the cell is eligible)
    baseline_pf        NUMERIC(6,3),
    baseline_n         INTEGER,
    baseline_as_of     DATE,
    monthly_trade_rate NUMERIC(6,2),

    notes              TEXT,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uq_auto_pause_eligibility UNIQUE (strategy, epic, config_set)
);

COMMENT ON TABLE auto_pause_eligibility IS
  'Auto-pause layer allowlist. A (strategy,epic,config_set) cell gets auto-pause '
  'protection ONLY if eligible=TRUE. baseline_* document the FROZEN promotion-time '
  'edge and must never be recomputed from recent performance.';

COMMENT ON COLUMN auto_pause_eligibility.baseline_pf IS
  'Frozen profit factor at promotion (established edge). Documentation only — '
  'the runtime never recomputes this; it gates membership, not the trip threshold.';

CREATE INDEX IF NOT EXISTS idx_auto_pause_eligibility_lookup
  ON auto_pause_eligibility (eligible, config_set);
