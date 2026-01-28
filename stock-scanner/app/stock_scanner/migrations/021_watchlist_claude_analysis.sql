-- Migration: 021_watchlist_claude_analysis.sql
-- Description: Store Claude AI analysis for technical watchlist entries

CREATE TABLE IF NOT EXISTS stock_watchlist_claude_analysis (
    id SERIAL PRIMARY KEY,
    watchlist_name VARCHAR(100) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    scan_date DATE,
    claude_grade VARCHAR(2),
    claude_score INTEGER,
    claude_conviction VARCHAR(10),
    claude_action VARCHAR(20),
    claude_thesis TEXT,
    claude_key_strengths TEXT[],
    claude_key_risks TEXT[],
    claude_position_rec VARCHAR(20),
    claude_stop_adjustment VARCHAR(10),
    claude_time_horizon VARCHAR(15),
    claude_raw_response TEXT,
    claude_analyzed_at TIMESTAMPTZ DEFAULT NOW(),
    claude_tokens_used INTEGER,
    claude_latency_ms INTEGER,
    claude_model VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_watchlist_claude UNIQUE (watchlist_name, ticker, scan_date)
);

CREATE INDEX IF NOT EXISTS idx_watchlist_claude_ticker
    ON stock_watchlist_claude_analysis(ticker, claude_analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_claude_watchlist
    ON stock_watchlist_claude_analysis(watchlist_name, claude_analyzed_at DESC);

-- Trigger for updated_at
DROP TRIGGER IF EXISTS update_watchlist_claude_timestamp ON stock_watchlist_claude_analysis;
CREATE TRIGGER update_watchlist_claude_timestamp
    BEFORE UPDATE ON stock_watchlist_claude_analysis
    FOR EACH ROW
    EXECUTE FUNCTION update_scanner_timestamp();
