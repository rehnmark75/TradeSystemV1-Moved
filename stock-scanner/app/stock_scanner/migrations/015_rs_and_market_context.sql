-- =============================================================================
-- STOCK SCANNER DATABASE SCHEMA - RELATIVE STRENGTH & MARKET CONTEXT
-- Migration: 015_rs_and_market_context.sql
-- Description: Add Relative Strength analysis and market regime tracking
-- =============================================================================

-- =============================================================================
-- ADD RS COLUMNS TO STOCK_SCREENING_METRICS
-- =============================================================================

-- Relative Strength vs SPY (market benchmark)
ALTER TABLE stock_screening_metrics
ADD COLUMN IF NOT EXISTS rs_vs_spy DECIMAL(8,4);  -- Ratio: stock 20d perf / SPY 20d perf

-- RS Percentile (rank among all stocks, 1-100)
ALTER TABLE stock_screening_metrics
ADD COLUMN IF NOT EXISTS rs_percentile INTEGER;

-- RS Trend (improving/stable/deteriorating based on 5-day RS change)
ALTER TABLE stock_screening_metrics
ADD COLUMN IF NOT EXISTS rs_trend VARCHAR(20);

-- Sector RS (stock performance vs its sector ETF)
ALTER TABLE stock_screening_metrics
ADD COLUMN IF NOT EXISTS sector_rs DECIMAL(8,4);

-- Comments
COMMENT ON COLUMN stock_screening_metrics.rs_vs_spy IS 'Relative Strength vs SPY (stock 20d return / SPY 20d return). >1 = outperforming market';
COMMENT ON COLUMN stock_screening_metrics.rs_percentile IS 'RS Percentile rank among all stocks (1-100). 90+ = top 10% performers';
COMMENT ON COLUMN stock_screening_metrics.rs_trend IS 'RS trend: improving, stable, deteriorating (based on 5-day RS change)';
COMMENT ON COLUMN stock_screening_metrics.sector_rs IS 'Relative Strength vs sector ETF';

-- Index for RS filtering and sorting
CREATE INDEX IF NOT EXISTS idx_metrics_rs_percentile
    ON stock_screening_metrics(calculation_date DESC, rs_percentile DESC NULLS LAST)
    WHERE rs_percentile IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_metrics_rs_spy
    ON stock_screening_metrics(calculation_date DESC, rs_vs_spy DESC NULLS LAST)
    WHERE rs_vs_spy IS NOT NULL;

-- =============================================================================
-- MARKET CONTEXT TABLE (daily market regime and breadth)
-- =============================================================================

CREATE TABLE IF NOT EXISTS market_context (
    id BIGSERIAL PRIMARY KEY,
    calculation_date DATE NOT NULL UNIQUE,

    -- Market Regime (based on SPY)
    market_regime VARCHAR(30) NOT NULL,  -- 'bull_confirmed', 'bull_weakening', 'bear_weakening', 'bear_confirmed'
    spy_price DECIMAL(12,4),
    spy_sma50 DECIMAL(12,4),
    spy_sma200 DECIMAL(12,4),
    spy_vs_sma50_pct DECIMAL(6,2),
    spy_vs_sma200_pct DECIMAL(6,2),
    spy_trend VARCHAR(20),  -- 'rising', 'falling', 'flat'

    -- Breadth Indicators
    pct_above_sma200 DECIMAL(5,2),  -- % of stocks above 200-day SMA
    pct_above_sma50 DECIMAL(5,2),   -- % of stocks above 50-day SMA
    pct_above_sma20 DECIMAL(5,2),   -- % of stocks above 20-day SMA

    -- New Highs / New Lows
    new_highs_count INTEGER,
    new_lows_count INTEGER,
    high_low_ratio DECIMAL(8,2),

    -- Advance/Decline
    advancing_count INTEGER,
    declining_count INTEGER,
    ad_ratio DECIMAL(8,2),

    -- Volatility (VIX proxy - we'll calculate from stock data)
    avg_atr_pct DECIMAL(5,2),
    volatility_regime VARCHAR(20),  -- 'low', 'normal', 'high', 'extreme'

    -- Strategy Recommendations based on regime
    recommended_strategies JSONB,  -- e.g., {"trend_following": 0.8, "mean_reversion": 0.2}

    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE market_context IS 'Daily market regime and breadth indicators for strategy selection';
COMMENT ON COLUMN market_context.market_regime IS 'Bull Confirmed (SPY > rising SMA200), Bull Weakening, Bear Weakening, Bear Confirmed';
COMMENT ON COLUMN market_context.pct_above_sma200 IS 'Market health indicator - % of stocks above 200-day SMA';
COMMENT ON COLUMN market_context.recommended_strategies IS 'Strategy weights based on current regime';

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_market_context_date
    ON market_context(calculation_date DESC);

-- =============================================================================
-- SECTOR ANALYSIS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS sector_analysis (
    id BIGSERIAL PRIMARY KEY,
    calculation_date DATE NOT NULL,
    sector VARCHAR(50) NOT NULL,
    sector_etf VARCHAR(10),  -- e.g., 'XLK' for Technology

    -- Sector Performance
    sector_return_1d DECIMAL(6,2),
    sector_return_5d DECIMAL(6,2),
    sector_return_20d DECIMAL(6,2),

    -- RS vs SPY
    rs_vs_spy DECIMAL(8,4),
    rs_percentile INTEGER,  -- Rank among 11 sectors (1-100 scale)
    rs_trend VARCHAR(20),   -- 'improving', 'stable', 'deteriorating'

    -- Sector Breadth
    stocks_in_sector INTEGER,
    pct_above_sma50 DECIMAL(5,2),
    pct_bullish_trend DECIMAL(5,2),

    -- Top Performers
    top_stocks JSONB,  -- Array of top 5 tickers with RS scores

    -- Stage Classification
    sector_stage VARCHAR(20),  -- 'leading', 'weakening', 'lagging', 'improving'

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_sector_date UNIQUE(calculation_date, sector)
);

COMMENT ON TABLE sector_analysis IS 'Daily sector rotation and relative strength analysis';
COMMENT ON COLUMN sector_analysis.sector_stage IS 'Sector rotation stage: leading (high RS, improving), weakening (high RS, declining), lagging (low RS, declining), improving (low RS, rising)';

-- Index for sector queries
CREATE INDEX IF NOT EXISTS idx_sector_analysis_date
    ON sector_analysis(calculation_date DESC, rs_vs_spy DESC);

CREATE INDEX IF NOT EXISTS idx_sector_analysis_sector
    ON sector_analysis(sector, calculation_date DESC);

-- =============================================================================
-- STOCK ALERTS TABLE (for real-time notifications)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_alerts (
    id BIGSERIAL PRIMARY KEY,

    -- Alert Info
    ticker VARCHAR(20) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,  -- 'breakout', 'watchlist_entry', 'volume_spike', 'rs_improvement', 'signal_triggered'
    alert_subtype VARCHAR(50),         -- e.g., 'resistance_break', 'ema50_cross', '3x_volume'

    -- Alert Data
    trigger_price DECIMAL(12,4),
    trigger_volume BIGINT,
    alert_message TEXT,
    alert_data JSONB,  -- Additional context (e.g., resistance level, indicator values)

    -- Severity/Priority
    priority VARCHAR(20) DEFAULT 'normal',  -- 'low', 'normal', 'high', 'critical'

    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'sent', 'acknowledged', 'expired'

    -- Delivery
    delivery_channels JSONB,  -- {"email": true, "push": true, "webhook": false}
    sent_at TIMESTAMP,

    -- Timestamps
    triggered_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE stock_alerts IS 'Real-time stock alerts for breakouts, watchlist entries, and signals';

-- Indexes for alert queries
CREATE INDEX IF NOT EXISTS idx_stock_alerts_ticker
    ON stock_alerts(ticker, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_stock_alerts_pending
    ON stock_alerts(status, triggered_at DESC)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_stock_alerts_type
    ON stock_alerts(alert_type, triggered_at DESC);

-- =============================================================================
-- CUSTOM SCREENS TABLE (user-defined screeners)
-- =============================================================================

CREATE TABLE IF NOT EXISTS custom_screens (
    id BIGSERIAL PRIMARY KEY,

    -- Screen Info
    screen_name VARCHAR(100) NOT NULL,
    description TEXT,

    -- Filter Criteria (stored as JSONB for flexibility)
    filters JSONB NOT NULL,  -- Array of filter conditions
    -- Example: [{"field": "rs_percentile", "operator": ">=", "value": 80}, {"field": "trend_strength", "operator": "in", "value": ["strong_up", "up"]}]

    -- Sort Configuration
    sort_by VARCHAR(50) DEFAULT 'rs_percentile',
    sort_order VARCHAR(10) DEFAULT 'DESC',

    -- Display Options
    display_columns JSONB,  -- Columns to show in results

    -- Schedule (optional auto-run)
    is_scheduled BOOLEAN DEFAULT FALSE,
    schedule_cron VARCHAR(50),  -- e.g., '0 6 * * 1-5' for weekdays at 6am

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP,
    last_result_count INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE custom_screens IS 'User-defined custom stock screeners with flexible filter criteria';

-- Index for active screens
CREATE INDEX IF NOT EXISTS idx_custom_screens_active
    ON custom_screens(is_active, screen_name);

-- =============================================================================
-- VIEWS FOR ENHANCED ANALYTICS
-- =============================================================================

-- View: Stocks with strong RS in strong sectors
CREATE OR REPLACE VIEW v_sector_leaders AS
SELECT
    m.ticker,
    i.name,
    i.sector,
    m.current_price,
    m.rs_vs_spy,
    m.rs_percentile,
    m.rs_trend,
    s.rs_vs_spy as sector_rs,
    s.rs_percentile as sector_rs_percentile,
    s.sector_stage,
    m.trend_strength,
    m.ma_alignment,
    m.atr_percent,
    m.price_change_20d
FROM stock_screening_metrics m
JOIN stock_instruments i ON m.ticker = i.ticker
LEFT JOIN sector_analysis s ON i.sector = s.sector
    AND s.calculation_date = m.calculation_date
WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
  AND m.rs_percentile >= 70
  AND s.rs_percentile >= 50
ORDER BY m.rs_percentile DESC, s.rs_percentile DESC;

COMMENT ON VIEW v_sector_leaders IS 'Top RS stocks in leading/improving sectors';

-- View: Market regime with recommended strategies
CREATE OR REPLACE VIEW v_current_market_regime AS
SELECT
    mc.calculation_date,
    mc.market_regime,
    mc.spy_price,
    mc.spy_vs_sma50_pct,
    mc.spy_vs_sma200_pct,
    mc.pct_above_sma200 as market_health,
    mc.pct_above_sma50 as intermediate_trend,
    mc.volatility_regime,
    mc.recommended_strategies,
    CASE
        WHEN mc.market_regime = 'bull_confirmed' THEN 'Favor trend-following, breakouts, momentum strategies'
        WHEN mc.market_regime = 'bull_weakening' THEN 'Reduce position sizes, tighten stops, avoid new breakouts'
        WHEN mc.market_regime = 'bear_weakening' THEN 'Watch for reversals, start small positions in leaders'
        WHEN mc.market_regime = 'bear_confirmed' THEN 'Favor mean reversion, avoid trend-following, stay defensive'
        ELSE 'Mixed signals - use caution'
    END as strategy_guidance
FROM market_context mc
WHERE mc.calculation_date = (SELECT MAX(calculation_date) FROM market_context);

COMMENT ON VIEW v_current_market_regime IS 'Current market regime with strategy recommendations';

-- =============================================================================
-- SECTOR ETF MAPPING (for sector RS calculations)
-- =============================================================================

CREATE TABLE IF NOT EXISTS sector_etf_mapping (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(50) NOT NULL UNIQUE,
    etf_ticker VARCHAR(10) NOT NULL,
    etf_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert sector ETF mappings (S&P 500 SPDR sector ETFs)
INSERT INTO sector_etf_mapping (sector, etf_ticker, etf_name) VALUES
    ('Technology', 'XLK', 'Technology Select Sector SPDR'),
    ('Health Care', 'XLV', 'Health Care Select Sector SPDR'),
    ('Financials', 'XLF', 'Financial Select Sector SPDR'),
    ('Consumer Discretionary', 'XLY', 'Consumer Discretionary Select Sector SPDR'),
    ('Communication Services', 'XLC', 'Communication Services Select Sector SPDR'),
    ('Industrials', 'XLI', 'Industrial Select Sector SPDR'),
    ('Consumer Staples', 'XLP', 'Consumer Staples Select Sector SPDR'),
    ('Energy', 'XLE', 'Energy Select Sector SPDR'),
    ('Utilities', 'XLU', 'Utilities Select Sector SPDR'),
    ('Real Estate', 'XLRE', 'Real Estate Select Sector SPDR'),
    ('Materials', 'XLB', 'Materials Select Sector SPDR')
ON CONFLICT (sector) DO NOTHING;

COMMENT ON TABLE sector_etf_mapping IS 'Mapping of sectors to their representative ETFs for RS calculations';
