-- Scan Performance Snapshot Table
-- Stores per-epic indicator data for EVERY scan cycle
-- Enables analysis of rejection patterns and correlation with signal quality
-- Created: 2026-01-05

-- Main scan performance snapshot table
CREATE TABLE IF NOT EXISTS scan_performance_snapshot (
    id SERIAL PRIMARY KEY,

    -- Scan identification (links to market_intelligence_history)
    scan_cycle_id VARCHAR(64) NOT NULL,
    scan_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    epic VARCHAR(50) NOT NULL,
    pair_name VARCHAR(20), -- e.g., EURUSD (derived from epic)

    -- Scan outcome for this epic
    signal_generated BOOLEAN NOT NULL DEFAULT FALSE,
    signal_type VARCHAR(10), -- 'buy', 'sell', NULL
    signal_id INTEGER, -- FK to alert_history if signal was generated
    rejection_reason VARCHAR(30), -- 'confidence', 'dedup', 'strategy_filter', 'smart_money', NULL
    rejection_details TEXT, -- Additional rejection context

    -- Confidence scores
    raw_confidence DECIMAL(5,4), -- Confidence before any filtering
    final_confidence DECIMAL(5,4), -- Confidence after processing (if signal passed)
    confidence_threshold DECIMAL(5,4), -- Threshold used for filtering

    -- Price data at scan time
    current_price DECIMAL(12,6),
    bid_price DECIMAL(12,6),
    ask_price DECIMAL(12,6),
    spread_pips DECIMAL(6,2),

    -- EMA Indicators (trend)
    ema_9 DECIMAL(12,6),
    ema_21 DECIMAL(12,6),
    ema_50 DECIMAL(12,6),
    ema_200 DECIMAL(12,6),
    ema_bias_4h VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
    price_vs_ema50 DECIMAL(8,4), -- percentage distance from 50 EMA

    -- MACD Indicators (momentum)
    macd_line DECIMAL(12,8),
    macd_signal DECIMAL(12,8),
    macd_histogram DECIMAL(12,8),
    macd_trend VARCHAR(20), -- 'bullish', 'bearish', 'neutral'

    -- RSI (momentum)
    rsi_14 DECIMAL(6,2),
    rsi_zone VARCHAR(20), -- 'oversold', 'neutral', 'overbought'

    -- Efficiency Ratio (Kaufman)
    efficiency_ratio DECIMAL(6,4),
    er_classification VARCHAR(20), -- 'strong_trend', 'weak_trend', 'choppy', 'very_choppy'

    -- ATR / Volatility
    atr_14 DECIMAL(12,6),
    atr_pips DECIMAL(8,2),
    atr_percentile DECIMAL(5,2),
    volatility_state VARCHAR(20), -- 'low', 'normal', 'high', 'extreme'

    -- Bollinger Bands
    bb_upper DECIMAL(12,6),
    bb_middle DECIMAL(12,6),
    bb_lower DECIMAL(12,6),
    bb_width DECIMAL(12,6),
    bb_width_percentile DECIMAL(5,2),
    bb_position VARCHAR(20), -- 'above_upper', 'upper_zone', 'middle', 'lower_zone', 'below_lower'

    -- ADX / Directional Strength
    adx DECIMAL(6,2),
    plus_di DECIMAL(6,2),
    minus_di DECIMAL(6,2),
    adx_trend_strength VARCHAR(20), -- 'no_trend', 'weak', 'moderate', 'strong', 'very_strong'

    -- Market context (from intelligence for this epic)
    market_regime VARCHAR(30), -- 'trending', 'ranging', 'breakout', 'reversal', 'high_volatility', 'low_volatility'
    regime_confidence DECIMAL(5,4),
    session VARCHAR(20), -- 'asian', 'london', 'new_york', 'overlap'
    session_volatility VARCHAR(20), -- 'low', 'medium', 'high', 'very_high'

    -- SMC Context (if available)
    near_order_block BOOLEAN DEFAULT FALSE,
    ob_type VARCHAR(10), -- 'bullish', 'bearish', NULL
    ob_distance_pips DECIMAL(8,2),
    near_fvg BOOLEAN DEFAULT FALSE,
    fvg_type VARCHAR(10), -- 'bullish', 'bearish', NULL
    fvg_distance_pips DECIMAL(8,2),
    liquidity_sweep_detected BOOLEAN DEFAULT FALSE,
    liquidity_sweep_type VARCHAR(20), -- 'buy_side', 'sell_side', NULL

    -- Smart Money analysis scores (if processed)
    smart_money_score DECIMAL(5,4),
    smart_money_validated BOOLEAN DEFAULT FALSE,

    -- Multi-timeframe confluence
    mtf_alignment VARCHAR(20), -- 'strong_bull', 'weak_bull', 'neutral', 'weak_bear', 'strong_bear'
    mtf_confluence_score DECIMAL(5,4),

    -- Entry quality assessment
    entry_quality_score DECIMAL(5,4),
    fib_zone_distance DECIMAL(8,4), -- Distance from optimal entry zone

    -- Extended indicator data (JSON for flexibility)
    extended_indicators JSONB,

    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES for optimal query performance
-- =============================================================================

-- Primary lookups
CREATE INDEX IF NOT EXISTS idx_sps_scan_cycle_id ON scan_performance_snapshot(scan_cycle_id);
CREATE INDEX IF NOT EXISTS idx_sps_scan_timestamp ON scan_performance_snapshot(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_sps_epic ON scan_performance_snapshot(epic);

-- Signal analysis indexes
CREATE INDEX IF NOT EXISTS idx_sps_signal_generated ON scan_performance_snapshot(signal_generated);
CREATE INDEX IF NOT EXISTS idx_sps_rejection_reason ON scan_performance_snapshot(rejection_reason);
CREATE INDEX IF NOT EXISTS idx_sps_signal_type ON scan_performance_snapshot(signal_type);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sps_epic_timestamp ON scan_performance_snapshot(epic, scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sps_rejection_timestamp ON scan_performance_snapshot(rejection_reason, scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sps_signal_regime ON scan_performance_snapshot(signal_generated, market_regime);
CREATE INDEX IF NOT EXISTS idx_sps_regime_session ON scan_performance_snapshot(market_regime, session);

-- Indicator-based indexes for performance analysis
CREATE INDEX IF NOT EXISTS idx_sps_efficiency_ratio ON scan_performance_snapshot(efficiency_ratio);
CREATE INDEX IF NOT EXISTS idx_sps_adx ON scan_performance_snapshot(adx);
CREATE INDEX IF NOT EXISTS idx_sps_rsi ON scan_performance_snapshot(rsi_14);
CREATE INDEX IF NOT EXISTS idx_sps_volatility ON scan_performance_snapshot(volatility_state);

-- Partial indexes for rejection analysis
CREATE INDEX IF NOT EXISTS idx_sps_rejected_confidence
    ON scan_performance_snapshot(scan_timestamp, raw_confidence)
    WHERE rejection_reason = 'confidence';

CREATE INDEX IF NOT EXISTS idx_sps_rejected_dedup
    ON scan_performance_snapshot(scan_timestamp, epic)
    WHERE rejection_reason = 'dedup';

CREATE INDEX IF NOT EXISTS idx_sps_signals_only
    ON scan_performance_snapshot(scan_timestamp, epic, signal_type)
    WHERE signal_generated = TRUE;

-- JSONB index for extended indicators
CREATE INDEX IF NOT EXISTS idx_sps_extended_indicators
    ON scan_performance_snapshot USING GIN(extended_indicators jsonb_path_ops);

-- =============================================================================
-- ANALYSIS VIEWS
-- =============================================================================

-- View: Rejection analysis by market conditions
CREATE OR REPLACE VIEW v_rejection_analysis AS
SELECT
    rejection_reason,
    market_regime,
    session,
    volatility_state,
    COUNT(*) as rejection_count,
    AVG(raw_confidence) as avg_raw_confidence,
    AVG(efficiency_ratio) as avg_efficiency_ratio,
    AVG(adx) as avg_adx,
    AVG(rsi_14) as avg_rsi,
    AVG(atr_percentile) as avg_atr_percentile
FROM scan_performance_snapshot
WHERE rejection_reason IS NOT NULL
GROUP BY rejection_reason, market_regime, session, volatility_state
ORDER BY rejection_count DESC;

-- View: Signal vs No-Signal indicator comparison
CREATE OR REPLACE VIEW v_signal_vs_nosignal_indicators AS
SELECT
    signal_generated,
    market_regime,
    COUNT(*) as scan_count,
    AVG(efficiency_ratio) as avg_er,
    AVG(adx) as avg_adx,
    AVG(rsi_14) as avg_rsi,
    AVG(atr_percentile) as avg_atr_pct,
    AVG(bb_width_percentile) as avg_bb_pct,
    AVG(CASE WHEN smart_money_validated THEN 1 ELSE 0 END) as sm_validation_rate
FROM scan_performance_snapshot
WHERE scan_timestamp > NOW() - INTERVAL '7 days'
GROUP BY signal_generated, market_regime
ORDER BY signal_generated DESC, scan_count DESC;

-- View: Hourly indicator averages by epic
CREATE OR REPLACE VIEW v_hourly_indicator_snapshot AS
SELECT
    DATE_TRUNC('hour', scan_timestamp) as hour,
    epic,
    COUNT(*) as scan_count,
    SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals_generated,
    SUM(CASE WHEN rejection_reason IS NOT NULL THEN 1 ELSE 0 END) as rejections,
    AVG(efficiency_ratio) as avg_er,
    AVG(adx) as avg_adx,
    AVG(atr_percentile) as avg_volatility,
    MODE() WITHIN GROUP (ORDER BY market_regime) as dominant_regime,
    MODE() WITHIN GROUP (ORDER BY volatility_state) as dominant_volatility
FROM scan_performance_snapshot
WHERE scan_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour, epic
ORDER BY hour DESC, epic;

-- View: Rejection patterns by time of day
CREATE OR REPLACE VIEW v_rejection_by_hour AS
SELECT
    EXTRACT(HOUR FROM scan_timestamp) as hour_of_day,
    session,
    rejection_reason,
    COUNT(*) as rejection_count,
    AVG(raw_confidence) as avg_confidence,
    AVG(efficiency_ratio) as avg_er
FROM scan_performance_snapshot
WHERE rejection_reason IS NOT NULL
  AND scan_timestamp > NOW() - INTERVAL '7 days'
GROUP BY hour_of_day, session, rejection_reason
ORDER BY hour_of_day, rejection_count DESC;

-- View: ER bands performance (correlate efficiency ratio with signal quality)
CREATE OR REPLACE VIEW v_er_band_performance AS
SELECT
    CASE
        WHEN efficiency_ratio >= 0.7 THEN 'strong_trend_0.7+'
        WHEN efficiency_ratio >= 0.5 THEN 'good_trend_0.5-0.7'
        WHEN efficiency_ratio >= 0.3 THEN 'weak_trend_0.3-0.5'
        ELSE 'choppy_<0.3'
    END as er_band,
    signal_generated,
    COUNT(*) as count,
    AVG(raw_confidence) as avg_confidence,
    AVG(adx) as avg_adx,
    AVG(rsi_14) as avg_rsi
FROM scan_performance_snapshot
WHERE scan_timestamp > NOW() - INTERVAL '7 days'
GROUP BY er_band, signal_generated
ORDER BY er_band, signal_generated DESC;

-- =============================================================================
-- RETENTION / CLEANUP
-- =============================================================================

-- Function to cleanup old snapshot records
CREATE OR REPLACE FUNCTION cleanup_old_scan_snapshots(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM scan_performance_snapshot
    WHERE scan_timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SAMPLE QUERIES
-- =============================================================================

/*
-- Get recent rejections with their indicator context
SELECT
    scan_timestamp, epic, rejection_reason, raw_confidence,
    efficiency_ratio, adx, rsi_14, volatility_state, market_regime
FROM scan_performance_snapshot
WHERE rejection_reason IS NOT NULL
ORDER BY scan_timestamp DESC
LIMIT 50;

-- Compare indicator distributions between signals and rejections
SELECT
    rejection_reason,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY efficiency_ratio) as median_er,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY adx) as median_adx,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY raw_confidence) as median_confidence
FROM scan_performance_snapshot
WHERE scan_timestamp > NOW() - INTERVAL '7 days'
GROUP BY rejection_reason;

-- Find optimal conditions for signals
SELECT
    market_regime, session, volatility_state,
    COUNT(*) as signal_count,
    AVG(final_confidence) as avg_confidence,
    AVG(efficiency_ratio) as avg_er
FROM scan_performance_snapshot
WHERE signal_generated = TRUE
  AND scan_timestamp > NOW() - INTERVAL '30 days'
GROUP BY market_regime, session, volatility_state
HAVING COUNT(*) >= 5
ORDER BY signal_count DESC;

-- Correlate with alert_history outcomes
SELECT
    s.market_regime, s.volatility_state, s.er_classification,
    COUNT(*) as signal_count,
    AVG(CASE WHEN a.claude_approved THEN 1 ELSE 0 END) as claude_approval_rate,
    AVG(a.claude_score) as avg_claude_score
FROM scan_performance_snapshot s
JOIN alert_history a ON s.signal_id = a.id
WHERE s.signal_generated = TRUE
  AND s.scan_timestamp > NOW() - INTERVAL '30 days'
GROUP BY s.market_regime, s.volatility_state, s.er_classification
HAVING COUNT(*) >= 3
ORDER BY claude_approval_rate DESC;
*/

-- Log table creation
DO $$
BEGIN
    RAISE NOTICE 'scan_performance_snapshot table created successfully with % indexes and % views',
        (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'scan_performance_snapshot'),
        4;
END $$;
