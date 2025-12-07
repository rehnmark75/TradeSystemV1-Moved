-- =============================================================================
-- STOCK SCANNER DATABASE SCHEMA - SCREENING & METRICS
-- Migration: 002_screening_metrics.sql
-- Description: Tables for daily screening metrics and watchlist management
-- =============================================================================

-- =============================================================================
-- SCREENING METRICS (calculated daily after market close)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_screening_metrics (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    calculation_date DATE NOT NULL,

    -- Current Price
    current_price DECIMAL(12,4),

    -- Volatility Metrics
    atr_14 DECIMAL(12,4),           -- 14-day ATR
    atr_percent DECIMAL(6,2),       -- ATR as % of price (key filter)
    historical_volatility_20 DECIMAL(6,2),  -- 20-day HV (annualized)

    -- Volume Metrics
    avg_volume_20 BIGINT,           -- 20-day average volume
    avg_dollar_volume DECIMAL(15,2), -- 20-day average dollar volume
    current_volume BIGINT,          -- Latest day volume
    relative_volume DECIMAL(8,2),   -- Current vs 20-day avg (RVol)

    -- Price Changes
    price_change_1d DECIMAL(6,2),   -- 1-day % change
    price_change_5d DECIMAL(6,2),   -- 5-day % change
    price_change_20d DECIMAL(6,2),  -- 20-day % change
    price_change_60d DECIMAL(6,2),  -- 60-day % change (3 months)

    -- Moving Averages
    sma_20 DECIMAL(12,4),
    sma_50 DECIMAL(12,4),
    sma_200 DECIMAL(12,4),
    ema_20 DECIMAL(12,4),

    -- Price vs MA (%)
    price_vs_sma20 DECIMAL(6,2),
    price_vs_sma50 DECIMAL(6,2),
    price_vs_sma200 DECIMAL(6,2),

    -- Trend Classification
    trend_strength VARCHAR(20),      -- 'strong_up', 'up', 'neutral', 'down', 'strong_down'
    ma_alignment VARCHAR(20),        -- 'bullish' (price>20>50>200), 'bearish', 'mixed'

    -- Momentum Indicators
    rsi_14 DECIMAL(6,2),
    macd DECIMAL(12,6),
    macd_signal DECIMAL(12,6),
    macd_histogram DECIMAL(12,6),

    -- Range Metrics
    daily_range_percent DECIMAL(6,2),  -- (High-Low)/Close %
    weekly_range_percent DECIMAL(6,2), -- 5-day range %

    -- Statistical
    z_score_50 DECIMAL(8,4),        -- Price z-score vs 50-day mean
    percentile_volume DECIMAL(5,2), -- Volume percentile (0-100)

    -- Metadata
    data_quality VARCHAR(20) DEFAULT 'good',  -- 'good', 'incomplete', 'stale'
    candles_available INTEGER,      -- Number of daily candles available

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_stock_metrics UNIQUE(ticker, calculation_date)
);

-- =============================================================================
-- STOCK WATCHLIST (tiered, scored stocks)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_watchlist (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    calculation_date DATE NOT NULL,

    -- Scoring (0-100)
    score DECIMAL(5,2) NOT NULL,    -- Composite score
    volume_score DECIMAL(5,2),      -- Volume component (0-30)
    volatility_score DECIMAL(5,2),  -- ATR/movement component (0-25)
    momentum_score DECIMAL(5,2),    -- Trend/momentum component (0-30)
    relative_strength_score DECIMAL(5,2), -- RS vs market (0-15)

    -- Tier (1=best, 4=monitoring)
    tier INTEGER NOT NULL,          -- 1, 2, 3, or 4
    rank_in_tier INTEGER,           -- Rank within tier
    rank_overall INTEGER,           -- Overall rank

    -- Key Metrics Snapshot
    current_price DECIMAL(12,4),
    atr_percent DECIMAL(6,2),
    avg_dollar_volume DECIMAL(15,2),
    relative_volume DECIMAL(8,2),
    price_change_20d DECIMAL(6,2),
    trend_strength VARCHAR(20),

    -- Flags
    is_new_to_tier BOOLEAN DEFAULT FALSE,  -- Just entered this tier
    tier_change INTEGER,            -- +1 = promoted, -1 = demoted, 0 = same

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_watchlist_entry UNIQUE(ticker, calculation_date),
    CONSTRAINT valid_tier CHECK (tier BETWEEN 1 AND 4)
);

-- =============================================================================
-- PIPELINE EXECUTION LOG
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_pipeline_log (
    id BIGSERIAL PRIMARY KEY,
    pipeline_name VARCHAR(50) NOT NULL,
    execution_date DATE NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds DECIMAL(10,2),

    -- Stage results
    results JSONB,

    -- Status
    status VARCHAR(20),  -- 'success', 'partial_failure', 'failed'
    error_message TEXT,

    CONSTRAINT unique_pipeline_run UNIQUE(pipeline_name, execution_date)
);

-- =============================================================================
-- ZERO-LAG MA SIGNALS (strategy-specific)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_zlma_signals (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    signal_timestamp TIMESTAMP NOT NULL,

    -- Signal
    signal_type VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL'

    -- Indicator Values
    zlma_value DECIMAL(12,4),
    ema_value DECIMAL(12,4),
    atr_value DECIMAL(12,4),

    -- Trend Levels (box bounds)
    level_top DECIMAL(12,4),
    level_bottom DECIMAL(12,4),

    -- Entry/Exit
    entry_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),

    -- Confidence
    confidence DECIMAL(5,2),

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_zlma_signal UNIQUE(ticker, signal_timestamp, signal_type)
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Screening metrics - most common queries
CREATE INDEX IF NOT EXISTS idx_metrics_date_ticker
    ON stock_screening_metrics(calculation_date DESC, ticker);

CREATE INDEX IF NOT EXISTS idx_metrics_atr
    ON stock_screening_metrics(calculation_date DESC, atr_percent DESC)
    WHERE atr_percent IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_metrics_volume
    ON stock_screening_metrics(calculation_date DESC, avg_dollar_volume DESC)
    WHERE avg_dollar_volume IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_metrics_rvol
    ON stock_screening_metrics(calculation_date DESC, relative_volume DESC)
    WHERE relative_volume > 1.0;

CREATE INDEX IF NOT EXISTS idx_metrics_trend
    ON stock_screening_metrics(calculation_date DESC, trend_strength)
    WHERE trend_strength IN ('strong_up', 'strong_down');

-- Watchlist indexes
CREATE INDEX IF NOT EXISTS idx_watchlist_date_tier
    ON stock_watchlist(calculation_date DESC, tier, rank_in_tier);

CREATE INDEX IF NOT EXISTS idx_watchlist_score
    ON stock_watchlist(calculation_date DESC, score DESC);

CREATE INDEX IF NOT EXISTS idx_watchlist_ticker
    ON stock_watchlist(ticker, calculation_date DESC);

-- Pipeline log
CREATE INDEX IF NOT EXISTS idx_pipeline_log_date
    ON stock_pipeline_log(execution_date DESC);

-- ZLMA signals
CREATE INDEX IF NOT EXISTS idx_zlma_signals_ticker_time
    ON stock_zlma_signals(ticker, signal_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_zlma_signals_recent
    ON stock_zlma_signals(signal_timestamp DESC)
    WHERE signal_timestamp > NOW() - INTERVAL '7 days';

-- =============================================================================
-- MATERIALIZED VIEW FOR DAILY CANDLES (fast access)
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS stock_daily_candles AS
SELECT
    ticker,
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    candles_used,
    created_at
FROM stock_candles_synthesized
WHERE timeframe = '1d'
ORDER BY ticker, timestamp DESC;

-- Unique index required for CONCURRENTLY refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_candles_ticker_time
    ON stock_daily_candles(ticker, timestamp);

CREATE INDEX IF NOT EXISTS idx_daily_candles_timestamp
    ON stock_daily_candles(timestamp DESC);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get latest watchlist
CREATE OR REPLACE FUNCTION get_watchlist(
    p_tier INTEGER DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE (
    rank INTEGER,
    ticker VARCHAR,
    score DECIMAL,
    tier INTEGER,
    price DECIMAL,
    atr_percent DECIMAL,
    dollar_volume DECIMAL,
    trend VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        w.rank_overall::INTEGER,
        w.ticker,
        w.score,
        w.tier,
        w.current_price,
        w.atr_percent,
        w.avg_dollar_volume,
        w.trend_strength
    FROM stock_watchlist w
    WHERE w.calculation_date = (
        SELECT MAX(calculation_date) FROM stock_watchlist
    )
    AND (p_tier IS NULL OR w.tier = p_tier)
    ORDER BY w.rank_overall
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to refresh daily candles view
CREATE OR REPLACE FUNCTION refresh_daily_candles_view()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY stock_daily_candles;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SCREENING VIEWS
-- =============================================================================

-- View: High volume movers (top candidates)
CREATE OR REPLACE VIEW stock_high_volume_movers AS
SELECT
    m.ticker,
    i.name,
    m.current_price,
    m.atr_percent,
    m.avg_dollar_volume / 1000000 as dollar_vol_millions,
    m.relative_volume,
    m.price_change_1d,
    m.price_change_5d,
    m.trend_strength,
    m.rsi_14,
    m.calculation_date
FROM stock_screening_metrics m
JOIN stock_instruments i ON m.ticker = i.ticker
WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
  AND m.avg_dollar_volume >= 10000000  -- $10M+ daily volume
  AND m.atr_percent >= 2.0              -- 2%+ ATR
  AND m.relative_volume >= 1.0          -- Above average volume
ORDER BY m.avg_dollar_volume DESC, m.atr_percent DESC
LIMIT 100;

-- View: Current watchlist summary
CREATE OR REPLACE VIEW stock_watchlist_summary AS
SELECT
    tier,
    COUNT(*) as stock_count,
    ROUND(AVG(score), 1) as avg_score,
    ROUND(AVG(atr_percent), 2) as avg_atr,
    ROUND(AVG(avg_dollar_volume) / 1000000, 1) as avg_dollar_vol_m,
    ROUND(AVG(price_change_20d), 2) as avg_20d_change
FROM stock_watchlist
WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
GROUP BY tier
ORDER BY tier;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE stock_screening_metrics IS 'Daily calculated metrics for stock screening (ATR, volume, momentum)';
COMMENT ON TABLE stock_watchlist IS 'Tiered watchlist with composite scores (Tier 1=best, Tier 4=monitoring)';
COMMENT ON TABLE stock_pipeline_log IS 'Execution log for data pipeline runs';
COMMENT ON TABLE stock_zlma_signals IS 'Signals from Zero-Lag MA Trend strategy';
COMMENT ON MATERIALIZED VIEW stock_daily_candles IS 'Fast-access view of synthesized daily candles';
