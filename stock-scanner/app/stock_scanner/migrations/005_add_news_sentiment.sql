-- Migration: Add News Sentiment Columns and Cache Table
-- Date: 2024-12-11
-- Description: Adds news sentiment analysis support for signal enrichment

-- ============================================================================
-- ADD NEWS SENTIMENT COLUMNS TO SIGNALS TABLE
-- ============================================================================

-- News sentiment score (-1.0 to 1.0)
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS news_sentiment_score DECIMAL(4, 3);

-- Sentiment level classification
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS news_sentiment_level VARCHAR(20);

-- Number of articles analyzed
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS news_headlines_count INT;

-- Contributing news factors for confluence display
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS news_factors TEXT[];

-- When news was last fetched/analyzed
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS news_analyzed_at TIMESTAMPTZ;

-- Add check constraint for sentiment level
ALTER TABLE stock_scanner_signals
DROP CONSTRAINT IF EXISTS valid_news_sentiment_level;

ALTER TABLE stock_scanner_signals
ADD CONSTRAINT valid_news_sentiment_level
CHECK (news_sentiment_level IS NULL OR news_sentiment_level IN (
    'very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish'
));

-- Add index for querying by sentiment
CREATE INDEX IF NOT EXISTS idx_scanner_signals_sentiment
ON stock_scanner_signals(news_sentiment_level)
WHERE news_sentiment_level IS NOT NULL;

-- ============================================================================
-- NEWS CACHE TABLE
-- ============================================================================
-- Caches fetched news articles to reduce API calls

CREATE TABLE IF NOT EXISTS stock_news_cache (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source VARCHAR(100),
    url TEXT,
    image_url TEXT,
    published_at TIMESTAMPTZ,
    category VARCHAR(50),

    -- Sentiment analysis results
    sentiment_score DECIMAL(4, 3),  -- VADER compound score (-1 to 1)

    -- Finnhub metadata
    finnhub_id BIGINT,

    -- Cache management
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '1 hour',

    -- Prevent duplicate articles
    CONSTRAINT unique_news_article UNIQUE (ticker, finnhub_id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_news_cache_ticker ON stock_news_cache(ticker);
CREATE INDEX IF NOT EXISTS idx_news_cache_published ON stock_news_cache(ticker, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_cache_expires ON stock_news_cache(expires_at);

-- Index for fetching recent news by ticker
CREATE INDEX IF NOT EXISTS idx_news_cache_recent
ON stock_news_cache(ticker, published_at DESC)
WHERE expires_at > NOW();

-- ============================================================================
-- NEWS FETCH LOG TABLE
-- ============================================================================
-- Tracks API calls for rate limiting and debugging

CREATE TABLE IF NOT EXISTS stock_news_fetch_log (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    fetch_timestamp TIMESTAMPTZ DEFAULT NOW(),
    articles_fetched INT DEFAULT 0,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    rate_limit_remaining INT,
    response_time_ms INT
);

-- Index for checking recent fetches (rate limiting)
CREATE INDEX IF NOT EXISTS idx_news_fetch_log_recent
ON stock_news_fetch_log(ticker, fetch_timestamp DESC);

-- Cleanup old logs after 7 days (can be done via scheduled job)
CREATE INDEX IF NOT EXISTS idx_news_fetch_log_cleanup
ON stock_news_fetch_log(fetch_timestamp)
WHERE fetch_timestamp < NOW() - INTERVAL '7 days';

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to check if news cache is valid for a ticker
CREATE OR REPLACE FUNCTION is_news_cache_valid(p_ticker VARCHAR, p_max_age_hours INT DEFAULT 1)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM stock_news_cache
        WHERE ticker = p_ticker
        AND fetched_at > NOW() - (p_max_age_hours || ' hours')::INTERVAL
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get cached news for a ticker
CREATE OR REPLACE FUNCTION get_cached_news(p_ticker VARCHAR, p_days INT DEFAULT 7)
RETURNS TABLE (
    headline TEXT,
    summary TEXT,
    source VARCHAR,
    url TEXT,
    published_at TIMESTAMPTZ,
    sentiment_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        nc.headline,
        nc.summary,
        nc.source,
        nc.url,
        nc.published_at,
        nc.sentiment_score
    FROM stock_news_cache nc
    WHERE nc.ticker = p_ticker
    AND nc.published_at > NOW() - (p_days || ' days')::INTERVAL
    AND nc.expires_at > NOW()
    ORDER BY nc.published_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired news cache
CREATE OR REPLACE FUNCTION cleanup_expired_news_cache()
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM stock_news_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for signals with news sentiment
CREATE OR REPLACE VIEW v_signals_with_news AS
SELECT
    s.id,
    s.signal_timestamp,
    s.scanner_name,
    s.ticker,
    s.signal_type,
    s.entry_price,
    s.composite_score,
    s.quality_tier,
    s.status,
    s.news_sentiment_score,
    s.news_sentiment_level,
    s.news_headlines_count,
    s.news_factors,
    s.news_analyzed_at,
    CASE
        WHEN s.news_sentiment_level IN ('very_bullish', 'bullish') AND s.signal_type = 'BUY' THEN 'Aligned'
        WHEN s.news_sentiment_level IN ('very_bearish', 'bearish') AND s.signal_type = 'SELL' THEN 'Aligned'
        WHEN s.news_sentiment_level = 'neutral' THEN 'Neutral'
        WHEN s.news_sentiment_level IS NULL THEN 'Not Analyzed'
        ELSE 'Contrary'
    END as news_signal_alignment
FROM stock_scanner_signals s
WHERE s.status = 'active'
ORDER BY s.signal_timestamp DESC;

-- View for news cache statistics
CREATE OR REPLACE VIEW v_news_cache_stats AS
SELECT
    ticker,
    COUNT(*) as cached_articles,
    MIN(published_at) as oldest_article,
    MAX(published_at) as newest_article,
    AVG(sentiment_score) as avg_sentiment,
    MAX(fetched_at) as last_fetch
FROM stock_news_cache
WHERE expires_at > NOW()
GROUP BY ticker
ORDER BY last_fetch DESC;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN stock_scanner_signals.news_sentiment_score IS 'VADER sentiment score from -1.0 (very bearish) to 1.0 (very bullish)';
COMMENT ON COLUMN stock_scanner_signals.news_sentiment_level IS 'Classified sentiment: very_bullish, bullish, neutral, bearish, very_bearish';
COMMENT ON COLUMN stock_scanner_signals.news_headlines_count IS 'Number of news articles analyzed for sentiment';
COMMENT ON COLUMN stock_scanner_signals.news_factors IS 'Contributing news factors for confluence display';
COMMENT ON COLUMN stock_scanner_signals.news_analyzed_at IS 'Timestamp when news sentiment was last analyzed';

COMMENT ON TABLE stock_news_cache IS 'Cache for Finnhub news articles to reduce API calls';
COMMENT ON TABLE stock_news_fetch_log IS 'Log of news API calls for rate limiting and debugging';
