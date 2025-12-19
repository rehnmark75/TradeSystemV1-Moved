-- ============================================================================
-- EMA DOUBLE CONFIRMATION REJECTIONS TABLE
-- ============================================================================
-- Purpose: Store rejection data from EMA Double Confirmation strategy for analysis
-- Created: 2025-12-19
-- Storage: ~0.5-1 MB/day (~200-400 rejections/day)
-- ============================================================================

-- Drop existing objects if they exist (for clean migration)
DROP VIEW IF EXISTS v_ema_double_near_misses;
DROP VIEW IF EXISTS v_ema_double_rejection_by_hour;
DROP VIEW IF EXISTS v_ema_double_rejection_by_session;
DROP VIEW IF EXISTS v_ema_double_rejection_by_stage;
DROP TABLE IF EXISTS ema_double_rejections;

-- ============================================================================
-- MAIN TABLE
-- ============================================================================

CREATE TABLE ema_double_rejections (
    id SERIAL PRIMARY KEY,
    scan_timestamp TIMESTAMP NOT NULL,
    epic VARCHAR(50) NOT NULL,
    pair VARCHAR(20) NOT NULL,

    -- Rejection Details
    rejection_stage VARCHAR(30) NOT NULL,  -- 'DATA_VALIDATION', 'NO_CROSSOVER', 'INSUFFICIENT_CONFIRMATIONS', 'SESSION_FILTER', 'HTF_TREND', 'FVG_FILTER', 'ADX_FILTER', 'RISK_VALIDATION', 'CONFIDENCE'
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

    -- EMA Context (Core Strategy Indicators)
    ema_fast_value DECIMAL(10,5),          -- EMA 9 value
    ema_slow_value DECIMAL(10,5),          -- EMA 21 value
    ema_trend_value DECIMAL(10,5),         -- EMA 200 value
    ema_fast_slow_separation_pips DECIMAL(8,2),  -- Distance between EMA 9 and 21

    -- HTF Trend Context (4H EMA 21)
    htf_ema_value DECIMAL(10,5),           -- 4H EMA 21 value
    htf_price_position VARCHAR(20),        -- 'above', 'below', 'at'
    htf_distance_pips DECIMAL(8,2),        -- Distance from 4H EMA

    -- Crossover History Context
    successful_crossover_count INTEGER,     -- Number of successful crossovers in lookback
    pending_crossover_count INTEGER,        -- Number of pending (unvalidated) crossovers
    last_crossover_direction VARCHAR(10),   -- Direction of last crossover
    last_crossover_bars_ago INTEGER,        -- Bars since last crossover

    -- ADX Context
    adx_value DECIMAL(6,2),
    adx_trending BOOLEAN,

    -- Volatility
    atr_15m DECIMAL(10,6),
    atr_pips DECIMAL(8,2),

    -- RSI Context
    rsi_value DECIMAL(5,2),

    -- Limit Order Context
    order_type VARCHAR(10),                -- 'market', 'limit'
    limit_offset_pips DECIMAL(6,2),
    limit_entry_price DECIMAL(10,5),

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

    -- OHLCV Snapshots (15m - primary timeframe)
    candle_15m_open DECIMAL(10,5),
    candle_15m_high DECIMAL(10,5),
    candle_15m_low DECIMAL(10,5),
    candle_15m_close DECIMAL(10,5),
    candle_15m_volume DECIMAL(15,2),

    -- OHLCV Snapshots (4H - HTF)
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
CREATE INDEX idx_ema_double_rej_timestamp ON ema_double_rejections(scan_timestamp);
CREATE INDEX idx_ema_double_rej_epic ON ema_double_rejections(epic);
CREATE INDEX idx_ema_double_rej_stage ON ema_double_rejections(rejection_stage);
CREATE INDEX idx_ema_double_rej_session ON ema_double_rejections(market_session);
CREATE INDEX idx_ema_double_rej_direction ON ema_double_rejections(attempted_direction);
CREATE INDEX idx_ema_double_rej_hour ON ema_double_rejections(market_hour);

-- Analysis patterns
CREATE INDEX idx_ema_double_rej_adx ON ema_double_rejections(adx_value);
CREATE INDEX idx_ema_double_rej_confidence ON ema_double_rejections(confidence_score);
CREATE INDEX idx_ema_double_rej_rr ON ema_double_rejections(potential_rr_ratio);
CREATE INDEX idx_ema_double_rej_crossover_count ON ema_double_rejections(successful_crossover_count);

-- Composite indexes for common queries
CREATE INDEX idx_ema_double_rej_epic_timestamp ON ema_double_rejections(epic, scan_timestamp DESC);
CREATE INDEX idx_ema_double_rej_stage_timestamp ON ema_double_rejections(rejection_stage, scan_timestamp DESC);

-- ============================================================================
-- ANALYSIS VIEWS
-- ============================================================================

-- View: Rejection counts by stage (last 30 days)
CREATE VIEW v_ema_double_rejection_by_stage AS
SELECT
    rejection_stage,
    COUNT(*) as rejection_count,
    COUNT(DISTINCT epic) as unique_pairs,
    ROUND(AVG(adx_value)::numeric, 2) as avg_adx,
    ROUND(AVG(spread_pips)::numeric, 2) as avg_spread,
    ROUND(AVG(successful_crossover_count)::numeric, 1) as avg_crossover_count
FROM ema_double_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY rejection_stage
ORDER BY rejection_count DESC;

-- View: Rejection patterns by session
CREATE VIEW v_ema_double_rejection_by_session AS
SELECT
    market_session,
    rejection_stage,
    COUNT(*) as rejection_count,
    ROUND(AVG(adx_value)::numeric, 2) as avg_adx
FROM ema_double_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY market_session, rejection_stage
ORDER BY market_session, rejection_count DESC;

-- View: Hourly rejection patterns
CREATE VIEW v_ema_double_rejection_by_hour AS
SELECT
    market_hour,
    rejection_stage,
    COUNT(*) as rejection_count
FROM ema_double_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY market_hour, rejection_stage
ORDER BY market_hour;

-- View: Near-miss signals (reached confidence stage but rejected)
CREATE VIEW v_ema_double_near_misses AS
SELECT
    scan_timestamp,
    epic,
    pair,
    attempted_direction,
    confidence_score,
    rejection_reason,
    potential_rr_ratio,
    market_session,
    htf_distance_pips,
    adx_value,
    successful_crossover_count
FROM ema_double_rejections
WHERE rejection_stage = 'CONFIDENCE'
  AND scan_timestamp >= NOW() - INTERVAL '30 days'
ORDER BY confidence_score DESC;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE ema_double_rejections IS 'Stores rejection data from EMA Double Confirmation strategy for analysis and improvement';
COMMENT ON COLUMN ema_double_rejections.rejection_stage IS 'Stage where rejection occurred: DATA_VALIDATION, NO_CROSSOVER, INSUFFICIENT_CONFIRMATIONS, SESSION_FILTER, HTF_TREND, FVG_FILTER, ADX_FILTER, RISK_VALIDATION, CONFIDENCE';
COMMENT ON COLUMN ema_double_rejections.rejection_details IS 'Full context as JSON for detailed analysis';
COMMENT ON COLUMN ema_double_rejections.confidence_breakdown IS 'Component scores that make up the confidence calculation';
COMMENT ON COLUMN ema_double_rejections.successful_crossover_count IS 'Number of validated successful crossovers in lookback window';

-- ============================================================================
-- GRANT PERMISSIONS (adjust as needed for your setup)
-- ============================================================================
-- GRANT SELECT, INSERT ON ema_double_rejections TO trading_app;
-- GRANT SELECT ON v_ema_double_rejection_by_stage TO trading_app;
-- GRANT SELECT ON v_ema_double_rejection_by_session TO trading_app;
-- GRANT SELECT ON v_ema_double_rejection_by_hour TO trading_app;
-- GRANT SELECT ON v_ema_double_near_misses TO trading_app;
