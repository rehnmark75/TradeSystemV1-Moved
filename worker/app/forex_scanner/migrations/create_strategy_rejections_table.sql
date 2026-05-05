-- Strategy Rejections Table
-- Generic rejection store for non-SMC strategies: MEAN_REVERSION, IMPULSE_FADE, XAU_GOLD.
-- SMC_SIMPLE keeps its own smc_simple_rejections table (richer schema, separate lifecycle).
--
-- Run against the strategy_config database:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/create_strategy_rejections_table.sql

CREATE TABLE IF NOT EXISTS strategy_rejections (
    id              BIGSERIAL PRIMARY KEY,
    strategy        VARCHAR(30)  NOT NULL,           -- MEAN_REVERSION / IMPULSE_FADE / XAU_GOLD
    epic            VARCHAR(50)  NOT NULL,
    pair            VARCHAR(20)  NOT NULL,
    scan_timestamp  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    stage           VARCHAR(50)  NOT NULL,           -- rejection stage code
    reason          TEXT,                            -- human-readable explanation
    direction       VARCHAR(10),                     -- BUY / SELL / NULL
    hour_utc        SMALLINT,                        -- 0-23, indexed for session analysis
    session         VARCHAR(20),                     -- london / overlap / new_york / asian
    details         JSONB                            -- strategy-specific context (ADX, ATR, body_pips…)
);

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sr_strategy          ON strategy_rejections (strategy);
CREATE INDEX IF NOT EXISTS idx_sr_epic              ON strategy_rejections (epic);
CREATE INDEX IF NOT EXISTS idx_sr_stage             ON strategy_rejections (stage);
CREATE INDEX IF NOT EXISTS idx_sr_hour              ON strategy_rejections (hour_utc);
CREATE INDEX IF NOT EXISTS idx_sr_ts                ON strategy_rejections (scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sr_strategy_epic_ts  ON strategy_rejections (strategy, epic, scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sr_strategy_stage    ON strategy_rejections (strategy, stage);

-- ── Analysis views ────────────────────────────────────────────────────────────

-- Rejection counts by strategy + stage + pair (last 30 days)
CREATE OR REPLACE VIEW v_sr_by_stage AS
SELECT
    strategy,
    epic,
    stage,
    session,
    COUNT(*)                                AS rejections,
    MAX(scan_timestamp)                     AS last_seen
FROM strategy_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY strategy, epic, stage, session
ORDER BY strategy, rejections DESC;

-- Hourly distribution per strategy (useful for session-filter tuning)
CREATE OR REPLACE VIEW v_sr_by_hour AS
SELECT
    strategy,
    epic,
    hour_utc,
    session,
    stage,
    COUNT(*) AS rejections
FROM strategy_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY strategy, epic, hour_utc, session, stage
ORDER BY strategy, hour_utc;

-- Top rejection stages per strategy over the last 7 days
CREATE OR REPLACE VIEW v_sr_top_stages AS
SELECT
    strategy,
    stage,
    COUNT(*)                                AS total,
    COUNT(DISTINCT epic)                    AS pairs_affected,
    ROUND(COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (PARTITION BY strategy), 1) AS pct_of_strategy
FROM strategy_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY strategy, stage
ORDER BY strategy, total DESC;
