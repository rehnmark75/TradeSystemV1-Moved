-- ============================================================================
-- monitor_only_outcomes — forward (counterfactual) outcomes for monitor-only signals
-- ============================================================================
-- One row per monitor-only alert_history signal. Populated nightly by
-- monitoring/monitor_outcome_analyzer.py, which walks ig_candles (1m) forward
-- from the signal and records how price actually behaved.
--
-- Design notes:
--   * The spine is MFE/MAE (max favorable / adverse excursion) — these require
--     NO stop/target and are valid for every strategy. They answer the
--     questions that matter most: did the signal move our way at all, did
--     losers go against us immediately (early_mae_pips), is TP too wide / SL
--     too tight.
--   * ref_* columns classify a single FIXED reference SL/TP grid so we also get
--     an exact HIT_TP/HIT_SL win-rate. This is a comparable diagnostic anchor,
--     NOT each strategy's native stop (native ATR/structure stops are not
--     recoverable from alert_history and are not reconstructed here).
--   * Lives in the `forex` DB alongside alert_history and ig_candles.
-- ============================================================================

CREATE TABLE IF NOT EXISTS monitor_only_outcomes (
    id                   SERIAL PRIMARY KEY,
    alert_id             INTEGER NOT NULL REFERENCES alert_history(id),
    strategy             VARCHAR(100),
    epic                 VARCHAR(50),
    pair                 VARCHAR(50),
    environment          VARCHAR(10),
    signal_timestamp     TIMESTAMP NOT NULL,
    direction            VARCHAR(10),          -- normalized: 'BUY' | 'SELL'
    entry_price          NUMERIC,
    pip_multiplier       INTEGER,              -- 10000 FX, 100 JPY, 10 gold
    horizon_minutes      INTEGER NOT NULL DEFAULT 1440,

    -- MFE/MAE spine (stop/target independent)
    mfe_pips             NUMERIC,              -- max favorable excursion over horizon
    mae_pips             NUMERIC,              -- max adverse excursion over horizon
    early_mae_pips       NUMERIC,              -- max adverse excursion in first 15 min
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

    UNIQUE (alert_id)
);

CREATE INDEX IF NOT EXISTS idx_moo_strategy        ON monitor_only_outcomes (strategy);
CREATE INDEX IF NOT EXISTS idx_moo_epic            ON monitor_only_outcomes (epic);
CREATE INDEX IF NOT EXISTS idx_moo_signal_ts       ON monitor_only_outcomes (signal_timestamp);
CREATE INDEX IF NOT EXISTS idx_moo_status          ON monitor_only_outcomes (status);
CREATE INDEX IF NOT EXISTS idx_moo_strategy_status ON monitor_only_outcomes (strategy, status);
