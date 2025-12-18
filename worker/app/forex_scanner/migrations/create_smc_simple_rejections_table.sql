-- ============================================================================
-- SMC SIMPLE REJECTIONS TABLE
-- ============================================================================
-- Purpose: Store rejection data from SMC Simple strategy for analysis
-- Created: 2025-12-18
-- Storage: ~1-2 MB/day (~500-900 rejections/day)
-- ============================================================================

-- Drop existing objects if they exist (for clean migration)
DROP VIEW IF EXISTS v_smc_near_misses;
DROP VIEW IF EXISTS v_smc_rejection_by_hour;
DROP VIEW IF EXISTS v_smc_rejection_by_session;
DROP VIEW IF EXISTS v_smc_rejection_by_stage;
DROP TABLE IF EXISTS smc_simple_rejections;

-- ============================================================================
-- MAIN TABLE
-- ============================================================================

CREATE TABLE smc_simple_rejections (
    id SERIAL PRIMARY KEY,
    scan_timestamp TIMESTAMP NOT NULL,
    epic VARCHAR(50) NOT NULL,
    pair VARCHAR(20) NOT NULL,

    -- Rejection Details
    rejection_stage VARCHAR(20) NOT NULL,  -- 'SESSION', 'COOLDOWN', 'TIER1_EMA', 'TIER2_SWING', 'TIER3_PULLBACK', 'RISK_LIMIT', 'RISK_RR', 'RISK_TP', 'CONFIDENCE'
    rejection_reason TEXT NOT NULL,
    rejection_details JSONB,

    -- Direction Attempted
    attempted_direction VARCHAR(10),  -- 'BULL', 'BEAR', or NULL

    -- Price Context
    current_price DECIMAL(10,5),
    bid_price DECIMAL(10,5),
    ask_price DECIMAL(10,5),
    spread_pips DECIMAL(5,2),

    -- Session Context
    market_hour INTEGER,
    market_session VARCHAR(20),
    is_market_hours BOOLEAN,

    -- EMA Context (Tier 1)
    ema_4h_value DECIMAL(10,5),
    ema_distance_pips DECIMAL(8,2),
    price_position_vs_ema VARCHAR(20),

    -- Volatility
    atr_15m DECIMAL(10,6),
    atr_5m DECIMAL(10,6),
    atr_percentile DECIMAL(5,2),

    -- Volume
    current_volume DECIMAL(15,2),
    volume_sma DECIMAL(15,2),
    volume_ratio DECIMAL(8,4),

    -- Swing Analysis (Tier 2)
    swing_high_level DECIMAL(10,5),
    swing_low_level DECIMAL(10,5),
    swing_lookback_bars INTEGER,
    swings_found_count INTEGER,
    last_swing_bars_ago INTEGER,

    -- Entry Zone Analysis (Tier 3)
    pullback_depth DECIMAL(6,4),
    fib_zone VARCHAR(20),
    swing_range_pips DECIMAL(8,2),

    -- Risk/Reward (if calculated)
    potential_entry DECIMAL(10,5),
    potential_stop_loss DECIMAL(10,5),
    potential_take_profit DECIMAL(10,5),
    potential_risk_pips DECIMAL(8,2),
    potential_reward_pips DECIMAL(8,2),
    potential_rr_ratio DECIMAL(6,3),

    -- Confidence (if calculated)
    confidence_score DECIMAL(5,4),
    confidence_breakdown JSONB,

    -- OHLCV Snapshots (5m)
    candle_5m_open DECIMAL(10,5),
    candle_5m_high DECIMAL(10,5),
    candle_5m_low DECIMAL(10,5),
    candle_5m_close DECIMAL(10,5),
    candle_5m_volume DECIMAL(15,2),

    -- OHLCV Snapshots (15m)
    candle_15m_open DECIMAL(10,5),
    candle_15m_high DECIMAL(10,5),
    candle_15m_low DECIMAL(10,5),
    candle_15m_close DECIMAL(10,5),
    candle_15m_volume DECIMAL(15,2),

    -- OHLCV Snapshots (4H)
    candle_4h_open DECIMAL(10,5),
    candle_4h_high DECIMAL(10,5),
    candle_4h_low DECIMAL(10,5),
    candle_4h_close DECIMAL(10,5),
    candle_4h_volume DECIMAL(15,2),

    -- Strategy Config
    strategy_version VARCHAR(20),
    strategy_config_hash VARCHAR(64),
    strategy_config JSONB,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Primary query patterns
CREATE INDEX idx_smc_rej_timestamp ON smc_simple_rejections(scan_timestamp);
CREATE INDEX idx_smc_rej_epic ON smc_simple_rejections(epic);
CREATE INDEX idx_smc_rej_stage ON smc_simple_rejections(rejection_stage);
CREATE INDEX idx_smc_rej_session ON smc_simple_rejections(market_session);
CREATE INDEX idx_smc_rej_direction ON smc_simple_rejections(attempted_direction);
CREATE INDEX idx_smc_rej_hour ON smc_simple_rejections(market_hour);

-- Analysis patterns
CREATE INDEX idx_smc_rej_atr ON smc_simple_rejections(atr_percentile);
CREATE INDEX idx_smc_rej_confidence ON smc_simple_rejections(confidence_score);
CREATE INDEX idx_smc_rej_rr ON smc_simple_rejections(potential_rr_ratio);

-- Composite indexes for common queries
CREATE INDEX idx_smc_rej_epic_timestamp ON smc_simple_rejections(epic, scan_timestamp DESC);
CREATE INDEX idx_smc_rej_stage_timestamp ON smc_simple_rejections(rejection_stage, scan_timestamp DESC);

-- ============================================================================
-- ANALYSIS VIEWS
-- ============================================================================

-- View: Rejection counts by stage (last 30 days)
CREATE VIEW v_smc_rejection_by_stage AS
SELECT
    rejection_stage,
    COUNT(*) as rejection_count,
    COUNT(DISTINCT epic) as unique_pairs,
    ROUND(AVG(atr_percentile)::numeric, 2) as avg_atr_percentile,
    ROUND(AVG(spread_pips)::numeric, 2) as avg_spread
FROM smc_simple_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY rejection_stage
ORDER BY rejection_count DESC;

-- View: Rejection patterns by session
CREATE VIEW v_smc_rejection_by_session AS
SELECT
    market_session,
    rejection_stage,
    COUNT(*) as rejection_count,
    ROUND(AVG(atr_percentile)::numeric, 2) as avg_volatility
FROM smc_simple_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY market_session, rejection_stage
ORDER BY market_session, rejection_count DESC;

-- View: Hourly rejection patterns
CREATE VIEW v_smc_rejection_by_hour AS
SELECT
    market_hour,
    rejection_stage,
    COUNT(*) as rejection_count
FROM smc_simple_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY market_hour, rejection_stage
ORDER BY market_hour;

-- View: Near-miss signals (reached confidence stage but rejected)
CREATE VIEW v_smc_near_misses AS
SELECT
    scan_timestamp,
    epic,
    pair,
    attempted_direction,
    confidence_score,
    rejection_reason,
    potential_rr_ratio,
    market_session,
    ema_distance_pips,
    pullback_depth,
    fib_zone
FROM smc_simple_rejections
WHERE rejection_stage = 'CONFIDENCE'
  AND scan_timestamp >= NOW() - INTERVAL '30 days'
ORDER BY confidence_score DESC;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE smc_simple_rejections IS 'Stores rejection data from SMC Simple strategy for analysis and improvement';
COMMENT ON COLUMN smc_simple_rejections.rejection_stage IS 'Stage where rejection occurred: SESSION, COOLDOWN, TIER1_EMA, TIER2_SWING, TIER3_PULLBACK, RISK_LIMIT, RISK_RR, RISK_TP, CONFIDENCE';
COMMENT ON COLUMN smc_simple_rejections.rejection_details IS 'Full context as JSON for detailed analysis';
COMMENT ON COLUMN smc_simple_rejections.confidence_breakdown IS 'Component scores that make up the confidence calculation';

-- ============================================================================
-- GRANT PERMISSIONS (adjust as needed for your setup)
-- ============================================================================
-- GRANT SELECT, INSERT ON smc_simple_rejections TO trading_app;
-- GRANT SELECT ON v_smc_rejection_by_stage TO trading_app;
-- GRANT SELECT ON v_smc_rejection_by_session TO trading_app;
-- GRANT SELECT ON v_smc_rejection_by_hour TO trading_app;
-- GRANT SELECT ON v_smc_near_misses TO trading_app;
