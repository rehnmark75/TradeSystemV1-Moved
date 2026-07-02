-- 043_scanner_cell_edge_router.sql
-- FOUNDATION for the per-cell "edge-map router" for the stock scanner system.
--
-- BIG PICTURE: ~20 stock scanners each fire across the whole 3000+ universe.
-- The router tags every signal with its market "character cell" (trend x vol x
-- liquidity x market_regime), then learns from CLEAN forward outcomes which
-- (scanner x cell) combos actually have edge, so that a later stage can route
-- only edge-positive cells into the tradable pool.
--
-- This migration is DATA-ONLY. It does NOT change any execution path, does not
-- touch live or demo trading, and does not itself gate any signal. It just adds
-- the columns that store the as-of cell classification on each signal, plus the
-- table the nightly analyzer writes the learned edge-map into.
--
-- Character-cell axes (thresholds are the source of truth in
--   stock_scanner/core/routing/cell_tagger.py -- keep in sync):
--   trend_state    : 'range' (adx<20) | 'mid' (20<=adx<25) | 'trend' (adx>=25)
--   vol_regime     : 'low' (atr%<2) | 'normal' (2<=atr%<4) | 'high' (atr%>=4)
--   liquidity_tier : 'thin' (rvol<1) | 'normal' (1<=rvol<3) | 'high' (rvol>=3)
--   market_regime  : copied as-of from market_context.market_regime
--
-- Cell values are computed AS-OF the signal date (causal: only metrics with
-- calculation_date <= signal_date), so backfilled and live-tagged rows are
-- directly comparable. All columns nullable -- a missing metric never blocks a
-- signal, it just leaves the cell fields NULL.
--
-- Idempotent: safe to re-run.

BEGIN;

-- (a) Per-signal character-cell columns on stock_scanner_signals -------------
ALTER TABLE stock_scanner_signals
    ADD COLUMN IF NOT EXISTS trend_state    VARCHAR(10),  -- 'range' | 'mid' | 'trend'
    ADD COLUMN IF NOT EXISTS vol_regime     VARCHAR(10),  -- 'low' | 'normal' | 'high'
    ADD COLUMN IF NOT EXISTS liquidity_tier VARCHAR(10),  -- 'thin' | 'normal' | 'high'
    ADD COLUMN IF NOT EXISTS cell_market_regime VARCHAR(50);  -- as-of market_context.market_regime

-- NOTE: stock_scanner_signals already has a `market_regime` column that scanners
-- populate at emission time from their own view of the tape. We deliberately do
-- NOT overwrite it. `cell_market_regime` is the router's own as-of snapshot from
-- market_context, so the edge-map's regime axis is consistent and causal
-- regardless of what each scanner happened to write.

-- Optional soft CHECK constraints (added only if not already present). These
-- allow NULL (missing metric) and only constrain non-null values.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_signal_trend_state') THEN
        ALTER TABLE stock_scanner_signals
            ADD CONSTRAINT chk_signal_trend_state
            CHECK (trend_state IS NULL OR trend_state IN ('range','mid','trend'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_signal_vol_regime') THEN
        ALTER TABLE stock_scanner_signals
            ADD CONSTRAINT chk_signal_vol_regime
            CHECK (vol_regime IS NULL OR vol_regime IN ('low','normal','high'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_signal_liquidity_tier') THEN
        ALTER TABLE stock_scanner_signals
            ADD CONSTRAINT chk_signal_liquidity_tier
            CHECK (liquidity_tier IS NULL OR liquidity_tier IN ('thin','normal','high'));
    END IF;
END$$;

-- Index to serve the analyzer's per-(scanner x cell) grouping and any future
-- route.ts lookups that filter by scanner + character cell.
CREATE INDEX IF NOT EXISTS idx_scanner_signals_cell
    ON stock_scanner_signals (scanner_name, trend_state, vol_regime);

-- (b) Learned edge-map table -------------------------------------------------
-- One row per (scanner_name x cell key). The cell key has up to four axes;
-- liquidity_tier and market_regime are NULLABLE so we can store both the
-- 2-axis grid (trend x vol, liquidity_tier & market_regime = NULL) and the
-- 3-axis grid (trend x vol x liquidity, market_regime = NULL) side by side and
-- compare whether the extra axis adds edge dispersion.
CREATE TABLE IF NOT EXISTS scanner_cell_edge (
    id             BIGSERIAL PRIMARY KEY,
    scanner_name   VARCHAR(100) NOT NULL,
    trend_state    VARCHAR(10)  NOT NULL,     -- always present in a cell key
    vol_regime     VARCHAR(10)  NOT NULL,     -- always present in a cell key
    liquidity_tier VARCHAR(10),               -- NULL => this row is a 2-axis rollup
    market_regime  VARCHAR(50),               -- NULL => regime axis not used for this row
    n              INTEGER      NOT NULL DEFAULT 0,
    wins           INTEGER      NOT NULL DEFAULT 0,
    pf             NUMERIC(10,4),
    win_rate       NUMERIC(6,4),
    avg_pnl_pct    NUMERIC(10,4),
    window_start   DATE,
    window_end     DATE,
    calendar_days  INTEGER,
    verdict        VARCHAR(15),               -- 'trade'|'monitor'|'block'|'insufficient'
    computed_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Uniqueness on the full cell key. COALESCE the nullable axes to sentinels so
-- that the 2-axis row (liquidity_tier NULL) and each 3-axis row are distinct and
-- upsertable. Postgres treats NULLs as distinct in a plain unique index, which
-- would allow duplicate 2-axis rows -- the expression index below prevents that.
CREATE UNIQUE INDEX IF NOT EXISTS uq_scanner_cell_edge_key
    ON scanner_cell_edge (
        scanner_name,
        trend_state,
        vol_regime,
        COALESCE(liquidity_tier, '__ALL__'),
        COALESCE(market_regime,  '__ALL__')
    );

CREATE INDEX IF NOT EXISTS idx_scanner_cell_edge_verdict
    ON scanner_cell_edge (scanner_name, verdict);

COMMIT;
