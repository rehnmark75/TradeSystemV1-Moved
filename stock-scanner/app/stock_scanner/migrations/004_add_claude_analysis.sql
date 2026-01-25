-- Migration: Add Claude AI Analysis Columns to Stock Scanner Signals
-- Version: 004
-- Description: Adds columns for storing Claude API analysis results
-- Date: 2024-12-08

-- =============================================================================
-- ADD CLAUDE ANALYSIS COLUMNS TO STOCK_SCANNER_SIGNALS
-- =============================================================================

-- Claude grade (A+, A, B, C, D)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_grade VARCHAR(2);

-- Claude numeric score (1-10)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_score INTEGER;

-- Conviction level (HIGH, MEDIUM, LOW)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_conviction VARCHAR(10);

-- Recommended action (STRONG BUY, BUY, HOLD, AVOID)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_action VARCHAR(15);

-- Investment thesis (2-3 sentence summary)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_thesis TEXT;

-- Key strengths identified
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_key_strengths TEXT[];

-- Key risks identified
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_key_risks TEXT[];

-- Position size recommendation (Full, Half, Quarter, Skip)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_position_rec VARCHAR(20);

-- Stop adjustment recommendation (Tighten, Keep, Widen)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_stop_adjustment VARCHAR(10);

-- Time horizon (Intraday, Swing, Position)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_time_horizon VARCHAR(15);

-- Full raw JSON response from Claude
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_raw_response TEXT;

-- Timestamp when Claude analysis was performed
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_analyzed_at TIMESTAMP;

-- API usage tracking - tokens used
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_tokens_used INTEGER;

-- Performance tracking - latency in milliseconds
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_latency_ms INTEGER;

-- Model used for analysis (e.g., claude-3-haiku, claude-3-sonnet)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS claude_model VARCHAR(50);

-- =============================================================================
-- CREATE INDEXES FOR CLAUDE ANALYSIS QUERIES
-- =============================================================================

-- Index for filtering by Claude grade (most common filter)
CREATE INDEX IF NOT EXISTS idx_scanner_signals_claude_grade
    ON stock_scanner_signals(claude_grade)
    WHERE claude_grade IS NOT NULL;

-- Index for finding unanalyzed signals
CREATE INDEX IF NOT EXISTS idx_scanner_signals_claude_pending
    ON stock_scanner_signals(signal_timestamp DESC)
    WHERE claude_grade IS NULL AND status = 'active';

-- Composite index for grade + score queries
CREATE INDEX IF NOT EXISTS idx_scanner_signals_claude_grade_score
    ON stock_scanner_signals(claude_grade, claude_score DESC)
    WHERE claude_grade IS NOT NULL;

-- =============================================================================
-- CREATE VIEW FOR SIGNALS WITH CLAUDE ANALYSIS
-- =============================================================================

CREATE OR REPLACE VIEW v_signals_with_claude AS
SELECT
    s.id,
    s.ticker,
    s.scanner_name,
    s.signal_type,
    s.signal_timestamp,
    s.entry_price,
    s.stop_loss,
    s.take_profit_1,
    s.take_profit_2,
    s.risk_percent,
    s.risk_reward_ratio,
    s.composite_score,
    s.quality_tier,
    s.trend_score,
    s.momentum_score,
    s.volume_score,
    s.pattern_score,
    s.confluence_score,
    s.confluence_factors,
    s.setup_description,
    s.market_regime,
    s.status,
    -- Claude analysis fields
    s.claude_grade,
    s.claude_score,
    s.claude_conviction,
    s.claude_action,
    s.claude_thesis,
    s.claude_key_strengths,
    s.claude_key_risks,
    s.claude_position_rec,
    s.claude_stop_adjustment,
    s.claude_time_horizon,
    s.claude_analyzed_at,
    s.claude_model,
    -- Related data
    i.name as company_name,
    i.sector,
    i.industry,
    i.earnings_date,
    i.institutional_percent,
    i.short_percent_float,
    i.analyst_rating,
    w.tier as watchlist_tier,
    w.score as watchlist_score
FROM stock_scanner_signals s
LEFT JOIN stock_instruments i ON s.ticker = i.ticker
LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
    AND w.calculation_date = (
        SELECT MAX(calculation_date)
        FROM stock_watchlist
        WHERE ticker = s.ticker
    )
WHERE s.claude_grade IS NOT NULL
ORDER BY
    CASE s.claude_grade
        WHEN 'A+' THEN 1
        WHEN 'A' THEN 2
        WHEN 'B' THEN 3
        WHEN 'C' THEN 4
        ELSE 5
    END,
    s.claude_score DESC,
    s.signal_timestamp DESC;

-- =============================================================================
-- CREATE VIEW FOR TODAY'S CLAUDE-ANALYZED SIGNALS
-- =============================================================================

CREATE OR REPLACE VIEW v_todays_claude_signals AS
SELECT
    s.ticker,
    s.scanner_name,
    s.signal_type,
    s.composite_score,
    s.quality_tier,
    s.claude_grade,
    s.claude_score,
    s.claude_conviction,
    s.claude_action,
    s.claude_thesis,
    s.claude_key_strengths,
    s.claude_key_risks,
    s.entry_price,
    s.stop_loss,
    s.take_profit_1,
    s.risk_reward_ratio,
    i.name as company_name,
    i.sector
FROM stock_scanner_signals s
LEFT JOIN stock_instruments i ON s.ticker = i.ticker
WHERE s.signal_timestamp >= CURRENT_DATE
    AND s.claude_grade IS NOT NULL
ORDER BY
    CASE s.claude_grade
        WHEN 'A+' THEN 1
        WHEN 'A' THEN 2
        WHEN 'B' THEN 3
        WHEN 'C' THEN 4
        ELSE 5
    END,
    s.claude_score DESC;

-- =============================================================================
-- CREATE FUNCTION TO GET CLAUDE ANALYSIS STATS
-- =============================================================================

CREATE OR REPLACE FUNCTION get_claude_analysis_stats(days_back INTEGER DEFAULT 7)
RETURNS TABLE (
    total_analyzed BIGINT,
    a_plus_count BIGINT,
    a_count BIGINT,
    b_count BIGINT,
    c_count BIGINT,
    d_count BIGINT,
    avg_score NUMERIC,
    high_conviction_count BIGINT,
    total_tokens_used BIGINT,
    avg_latency_ms NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_analyzed,
        COUNT(*) FILTER (WHERE claude_grade = 'A+')::BIGINT as a_plus_count,
        COUNT(*) FILTER (WHERE claude_grade = 'A')::BIGINT as a_count,
        COUNT(*) FILTER (WHERE claude_grade = 'B')::BIGINT as b_count,
        COUNT(*) FILTER (WHERE claude_grade = 'C')::BIGINT as c_count,
        COUNT(*) FILTER (WHERE claude_grade = 'D')::BIGINT as d_count,
        ROUND(AVG(claude_score)::NUMERIC, 2) as avg_score,
        COUNT(*) FILTER (WHERE claude_conviction = 'HIGH')::BIGINT as high_conviction_count,
        COALESCE(SUM(claude_tokens_used), 0)::BIGINT as total_tokens_used,
        ROUND(AVG(claude_latency_ms)::NUMERIC, 0) as avg_latency_ms
    FROM stock_scanner_signals
    WHERE claude_analyzed_at >= CURRENT_DATE - days_back
        AND claude_grade IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON COLUMN stock_scanner_signals.claude_grade IS 'AI-assigned quality grade: A+ (exceptional), A (strong), B (moderate), C (weak), D (avoid)';
COMMENT ON COLUMN stock_scanner_signals.claude_score IS 'Numeric score from 1-10 assigned by Claude';
COMMENT ON COLUMN stock_scanner_signals.claude_conviction IS 'Conviction level: HIGH, MEDIUM, LOW';
COMMENT ON COLUMN stock_scanner_signals.claude_action IS 'Recommended action: STRONG BUY, BUY, HOLD, AVOID';
COMMENT ON COLUMN stock_scanner_signals.claude_thesis IS 'Investment thesis explaining the AI reasoning';
COMMENT ON COLUMN stock_scanner_signals.claude_key_strengths IS 'Array of key strengths identified by AI';
COMMENT ON COLUMN stock_scanner_signals.claude_key_risks IS 'Array of key risks identified by AI';
COMMENT ON COLUMN stock_scanner_signals.claude_position_rec IS 'Position size recommendation: Full, Half, Quarter, Skip';
COMMENT ON COLUMN stock_scanner_signals.claude_analyzed_at IS 'Timestamp when Claude analysis was performed';
COMMENT ON COLUMN stock_scanner_signals.claude_tokens_used IS 'Number of API tokens used for this analysis';
COMMENT ON COLUMN stock_scanner_signals.claude_latency_ms IS 'API response time in milliseconds';
