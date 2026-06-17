-- 038_add_dbc_pattern_monitor_columns.sql
-- Monitor-only logging of the "decline -> base -> climb" (rounding-bottom)
-- daily pattern flag on scanner signals. Computed AS-OF signal_date (no
-- look-ahead) at save time. ADDITIVE + NULLABLE: NOT read by candidate
-- selection (route.ts candidate_score) — pure forward-data collection to
-- validate the in-sample gap_and_go confluence lift (PF 0.89 -> 1.21).
--
-- Populated only for scanners in DBC_MONITOR_SCANNERS (currently gap_and_go);
-- NULL for all others. See patterns/decline_base_climb.py and
-- analysis/pattern_overlap_edge.py.

ALTER TABLE stock_scanner_signals
    ADD COLUMN IF NOT EXISTS dbc_pattern_present BOOLEAN,
    ADD COLUMN IF NOT EXISTS dbc_pattern_score   NUMERIC(4,2);

COMMENT ON COLUMN stock_scanner_signals.dbc_pattern_present IS
    'Monitor-only: decline->base->climb pattern present as-of signal_date (no look-ahead). NULL = not evaluated for this scanner.';
COMMENT ON COLUMN stock_scanner_signals.dbc_pattern_score IS
    'Monitor-only: decline->base->climb fit score [0..6] when present; NULL otherwise.';
