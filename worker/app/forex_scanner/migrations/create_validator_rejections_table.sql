-- Migration: create_validator_rejections_table.sql
-- Purpose: Persist all TradeValidator rejection decisions for UI analysis
--          Previously only Claude rejections and reversal filter were stored;
--          all other rejection steps (confidence, R:R, LPF, market intel, etc.)
--          were silently dropped to Python logs only.
-- Database: forex

DROP TABLE IF EXISTS validator_rejections;

CREATE TABLE validator_rejections (
    id                  SERIAL PRIMARY KEY,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    epic                VARCHAR(100) NOT NULL,
    pair                VARCHAR(20)  NOT NULL,
    signal_type         VARCHAR(10),                    -- BULL / BEAR
    strategy            VARCHAR(50),
    confidence_score    DECIMAL(6,4),
    step                VARCHAR(30)  NOT NULL,          -- see step values below
    rejection_reason    TEXT         NOT NULL,

    -- Price context (populated when available)
    entry_price         DECIMAL(12,6),
    risk_pips           DECIMAL(8,2),
    reward_pips         DECIMAL(8,2),
    rr_ratio            DECIMAL(6,3),

    -- Market context at rejection time (from signal dict)
    market_regime       VARCHAR(50),
    market_session      VARCHAR(30),

    -- LPF-specific (populated only for step = 'LPF')
    lpf_penalty         DECIMAL(5,2),
    lpf_would_block     BOOLEAN,
    lpf_triggered_rules JSONB
);

-- step values (maps to validator step names):
--   STRUCTURE          - required field / format validation
--   MARKET_HOURS       - outside trading hours
--   EPIC               - epic blocked or invalid
--   CONFIDENCE         - below minimum confidence threshold
--   RISK               - R:R ratio or risk % violation
--   SR_LEVEL           - S/R blocks path to target
--   NEWS               - high-impact news filter
--   MARKET_INTELLIGENCE - regime / market bias / reversal filter
--   TRADING_SUITABILITY - volatility / liquidity / spread
--   LPF                - Loss Prevention Filter hard block
--   CLAUDE             - Claude AI rejected or low score

CREATE INDEX idx_vr_created_at   ON validator_rejections(created_at DESC);
CREATE INDEX idx_vr_epic         ON validator_rejections(epic);
CREATE INDEX idx_vr_step         ON validator_rejections(step);
CREATE INDEX idx_vr_pair_step    ON validator_rejections(pair, step);
CREATE INDEX idx_vr_signal_type  ON validator_rejections(signal_type);

-- Convenience view: rejection counts by step and pair (last 30 days)
CREATE OR REPLACE VIEW v_validator_rejection_summary AS
SELECT
    pair,
    step,
    signal_type,
    COUNT(*)                            AS total,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days')   AS last_7d,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day')    AS last_24h,
    AVG(confidence_score)               AS avg_confidence,
    AVG(rr_ratio)                       AS avg_rr_ratio,
    MAX(created_at)                     AS last_seen
FROM validator_rejections
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY pair, step, signal_type
ORDER BY pair, total DESC;
