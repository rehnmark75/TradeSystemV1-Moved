-- ============================================================================
-- Auto-Pause Layer — shadow trip source (Trip Rule B) eligibility columns
-- Database: strategy_config
-- ============================================================================
-- trade_log is too sparse for the rolling trade-based trip rule (Rule A) on
-- most cells (~<=3 closed trades / cell / 30d). Rule B evaluates the ref-grid
-- shadow series from monitor_only_outcomes instead (~40-170 outcomes / cell /
-- 30d). trip_source picks the rule(s) per cell; baseline_shadow_* freeze the
-- enrollment-time shadow edge (same doctrine as baseline_pf: NEVER recomputed
-- from recent data — the WR-drop leg of Rule B measures decay against it).
--
-- Apply:
--   docker exec -i postgres psql -U postgres -d strategy_config \
--     < worker/app/forex_scanner/migrations/add_auto_pause_shadow_columns.sql

ALTER TABLE auto_pause_eligibility
    ADD COLUMN IF NOT EXISTS trip_source        VARCHAR(8)   NOT NULL DEFAULT 'trades',
    ADD COLUMN IF NOT EXISTS baseline_shadow_pf NUMERIC(6,3),
    ADD COLUMN IF NOT EXISTS baseline_shadow_wr NUMERIC(5,4),
    ADD COLUMN IF NOT EXISTS baseline_shadow_n  INTEGER,
    ADD COLUMN IF NOT EXISTS baseline_source    TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_auto_pause_trip_source'
    ) THEN
        ALTER TABLE auto_pause_eligibility
            ADD CONSTRAINT chk_auto_pause_trip_source
            CHECK (trip_source IN ('trades', 'shadow', 'both'));
    END IF;
END $$;

COMMENT ON COLUMN auto_pause_eligibility.trip_source IS
  'Which trip rule protects this cell: trades (Rule A, trade_log PF), shadow '
  '(Rule B, monitor_only_outcomes ref-grid), or both (either rule may trip).';

COMMENT ON COLUMN auto_pause_eligibility.baseline_shadow_wr IS
  'FROZEN enrollment-time ref-grid win rate (0-1). Rule B trips when the '
  'rolling shadow WR falls shadow_trip_wr_drop below this. Never recompute '
  'from recent data.';

COMMENT ON COLUMN auto_pause_eligibility.baseline_source IS
  'Where the shadow baseline came from, e.g. "monitor_only_outcomes '
  '2026-04-01..2026-05-15 (n=88)" or "90d backtest 2026-07-01".';
