-- ============================================================================
-- rejection_outcomes — forward (counterfactual) outcomes for GATED-but-real setups
-- ============================================================================
-- Sibling of monitor_only_outcomes, but for signals a strategy *rejected* at an
-- edge gate (ADX ceiling, ER floor, confidence floor, cooldown, …) instead of
-- emitting. These never reach alert_history; they land in strategy_rejections
-- (strategy_config DB). One row here per evaluable rejection.
--
-- Purpose: per-GATE edge scoring. For each gate we can now ask "what did the
-- setups it blocked actually do?" — if the blocked population loses (PF < 1) the
-- gate earns its keep; if it wins (PF > ~1.3) the gate is costing edge AND
-- starving the monitor phase of the data needed to promote an epic.
--
-- Only rejections with a real trigger are evaluated: the analyzer selects
-- strategy_rejections rows where direction IS NOT NULL, which (by construction
-- in the strategies) are exactly the rejections that fired AFTER a directional
-- setup was established. Structural rejections (no_trigger, insufficient_data,
-- session_blocked, …) carry direction NULL and have no tradeable counterfactual.
--
-- Simulation reuses monitoring/monitor_outcome_analyzer.MonitorOutcomeAnalyzer.
-- simulate() verbatim: entry is reconstructed from the first post-signal 1m
-- candle open (rejections store no price), then MFE/MAE + a fixed reference
-- SL/TP grid are walked forward over a 24h horizon from ig_candles.
--
-- Lives in the `forex` DB (alongside ig_candles); keyed on the source
-- strategy_rejections.id (stored as rejection_id; no cross-DB FK).
--
--   docker exec postgres psql -U postgres -d forex \
--     -f /app/forex_scanner/migrations/create_rejection_outcomes.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS rejection_outcomes (
    id                   SERIAL PRIMARY KEY,
    rejection_id         BIGINT NOT NULL,      -- strategy_rejections.id (other DB; no FK)
    strategy             VARCHAR(100),
    epic                 VARCHAR(50),
    pair                 VARCHAR(50),
    stage                VARCHAR(50),          -- the GATE that blocked it (rejection stage)
    reason               TEXT,
    session              VARCHAR(20),
    hour_utc             SMALLINT,
    signal_timestamp     TIMESTAMP NOT NULL,
    direction            VARCHAR(10),          -- normalized: 'BUY' | 'SELL'
    entry_price          NUMERIC,
    pip_multiplier       INTEGER,              -- 10000 FX, 100 JPY, 10 gold
    horizon_minutes      INTEGER NOT NULL DEFAULT 1440,

    -- MFE/MAE spine (stop/target independent)
    mfe_pips             NUMERIC,
    mae_pips             NUMERIC,
    early_mae_pips       NUMERIC,
    time_to_mfe_minutes  INTEGER,
    time_to_mae_minutes  INTEGER,

    -- signed net move (favorable = positive) at each horizon boundary
    pnl_60m_pips         NUMERIC,
    pnl_240m_pips        NUMERIC,
    pnl_1440m_pips       NUMERIC,

    -- fixed reference SL/TP grid outcome (exact win-rate anchor)
    ref_sl_pips          NUMERIC,
    ref_tp_pips          NUMERIC,
    ref_outcome          VARCHAR(20),          -- 'HIT_TP' | 'HIT_SL' | 'TIMEOUT'
    ref_pnl_pips         NUMERIC,              -- realized pips under ref grid
    time_to_tp_minutes   INTEGER,
    time_to_sl_minutes   INTEGER,

    -- bookkeeping
    candles_evaluated    INTEGER,
    status               VARCHAR(20),          -- 'RESOLVED' | 'OPEN' | 'NO_DATA'
    evaluated_until      TIMESTAMP,
    created_at           TIMESTAMP DEFAULT now(),
    updated_at           TIMESTAMP DEFAULT now(),

    UNIQUE (rejection_id)
);

CREATE INDEX IF NOT EXISTS idx_ro_strategy        ON rejection_outcomes (strategy);
CREATE INDEX IF NOT EXISTS idx_ro_epic            ON rejection_outcomes (epic);
CREATE INDEX IF NOT EXISTS idx_ro_stage           ON rejection_outcomes (stage);
CREATE INDEX IF NOT EXISTS idx_ro_signal_ts       ON rejection_outcomes (signal_timestamp);
CREATE INDEX IF NOT EXISTS idx_ro_status          ON rejection_outcomes (status);
CREATE INDEX IF NOT EXISTS idx_ro_strategy_stage  ON rejection_outcomes (strategy, stage);

-- ── Per-gate edge view ────────────────────────────────────────────────────────
-- One row per (strategy, gate, epic, direction) over RESOLVED rejections, with
-- the PF/win-rate of the BLOCKED population under the fixed reference grid.
--
-- Reading it:
--   blocked_pf < ~1.0  → gate blocks net losers → it is earning its keep (KEEP).
--   blocked_pf > ~1.3  → gate blocks net winners → it is costing edge and data;
--                        candidate to loosen for the monitor phase (REVIEW).
--   n_resolved small   → not yet decidable; this is the data-starvation signal.
-- ref grid PF only — a comparable anchor, NOT the strategy's native SL/TP.
CREATE OR REPLACE VIEW v_rejection_gate_edge AS
SELECT
    strategy,
    stage                                                          AS gate,
    epic,
    direction,
    COUNT(*)                                                       AS n_resolved,
    ROUND(AVG((ref_outcome = 'HIT_TP')::int) * 100.0, 1)           AS blocked_win_rate_pct,
    ROUND(
        COALESCE(SUM(ref_pnl_pips) FILTER (WHERE ref_pnl_pips > 0), 0)
        / NULLIF(ABS(SUM(ref_pnl_pips) FILTER (WHERE ref_pnl_pips < 0)), 0),
        2
    )                                                              AS blocked_pf,
    ROUND(AVG(mfe_pips), 1)                                        AS avg_mfe_pips,
    ROUND(AVG(mae_pips), 1)                                        AS avg_mae_pips,
    ROUND(SUM(ref_pnl_pips), 1)                                    AS blocked_net_pips,
    MAX(signal_timestamp)                                          AS last_seen
FROM rejection_outcomes
WHERE status = 'RESOLVED'
GROUP BY strategy, stage, epic, direction
ORDER BY strategy, gate, epic, direction;
