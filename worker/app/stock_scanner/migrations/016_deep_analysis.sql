-- Migration: 016_deep_analysis.sql
-- Description: Create stock_deep_analysis table for storing Deep Analysis Quality (DAQ) scores
-- Date: 2026-01-12

-- Create the deep analysis table
CREATE TABLE IF NOT EXISTS stock_deep_analysis (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES stock_scanner_signals(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    analysis_timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Deep Analysis Quality Score (0-100)
    daq_score INTEGER CHECK (daq_score >= 0 AND daq_score <= 100),
    daq_grade VARCHAR(2) CHECK (daq_grade IN ('A+', 'A', 'B', 'C', 'D')),

    -- Technical Deep Scores (each 0-100, weighted to final DAQ)
    mtf_score INTEGER CHECK (mtf_score >= 0 AND mtf_score <= 100),        -- Multi-timeframe confluence
    volume_score INTEGER CHECK (volume_score >= 0 AND volume_score <= 100), -- Volume analysis
    smc_score INTEGER CHECK (smc_score >= 0 AND smc_score <= 100),        -- Smart Money Concepts

    -- Fundamental Deep Scores (each 0-100)
    quality_score INTEGER CHECK (quality_score >= 0 AND quality_score <= 100),   -- Financial health (ROE, margins, D/E)
    catalyst_score INTEGER CHECK (catalyst_score >= 0 AND catalyst_score <= 100), -- Earnings/event timing risk (inverted)
    institutional_score INTEGER CHECK (institutional_score >= 0 AND institutional_score <= 100), -- Institutional ownership

    -- Contextual Scores (each 0-100)
    news_score INTEGER CHECK (news_score >= 0 AND news_score <= 100),     -- News sentiment
    regime_score INTEGER CHECK (regime_score >= 0 AND regime_score <= 100), -- Market regime alignment
    sector_score INTEGER CHECK (sector_score >= 0 AND sector_score <= 100), -- Sector rotation

    -- Risk Flags
    earnings_within_7d BOOLEAN DEFAULT FALSE,
    high_short_interest BOOLEAN DEFAULT FALSE,   -- short_percent_float > 20%
    low_liquidity BOOLEAN DEFAULT FALSE,         -- avg volume < threshold
    extreme_volatility BOOLEAN DEFAULT FALSE,    -- ATR% > threshold
    sector_underperforming BOOLEAN DEFAULT FALSE,

    -- Multi-Timeframe Details (JSON for flexibility)
    mtf_details JSONB,  -- {1h: {trend: 'bullish', ema_aligned: true}, 4h: {...}, 1d: {...}}

    -- Volume Profile Details
    volume_details JSONB,  -- {relative_volume: 1.5, accumulation: true, unusual: false}

    -- SMC Details (from stock_screening_metrics)
    smc_details JSONB,  -- {trend: 'bullish', last_bos: {...}, nearest_ob: {...}}

    -- Fundamental Details
    fundamental_details JSONB,  -- {roe: 145, profit_margin: 25, debt_equity: 1.8, ...}

    -- Market Context Details
    context_details JSONB,  -- {regime: 'bullish', spy_trend: 'up', sector_rs: 1.2, ...}

    -- News Summary
    news_summary TEXT,
    news_articles_count INTEGER DEFAULT 0,
    top_headlines JSONB,  -- [{headline: '...', sentiment: 0.5}, ...]

    -- Claude Synthesis (populated separately via manual trigger)
    claude_synthesis TEXT,
    claude_action VARCHAR(20),
    claude_analyzed_at TIMESTAMPTZ,

    -- Processing Metadata
    analysis_duration_ms INTEGER,
    components_analyzed TEXT[],  -- ['mtf', 'volume', 'smc', 'fundamental', 'news', 'regime', 'sector']
    errors TEXT[],

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_deep_analysis_signal_id ON stock_deep_analysis(signal_id);
CREATE INDEX IF NOT EXISTS idx_deep_analysis_ticker ON stock_deep_analysis(ticker);
CREATE INDEX IF NOT EXISTS idx_deep_analysis_daq_score ON stock_deep_analysis(daq_score DESC);
CREATE INDEX IF NOT EXISTS idx_deep_analysis_daq_grade ON stock_deep_analysis(daq_grade);
CREATE INDEX IF NOT EXISTS idx_deep_analysis_timestamp ON stock_deep_analysis(analysis_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_deep_analysis_created_at ON stock_deep_analysis(created_at DESC);

-- Unique constraint: one deep analysis per signal
CREATE UNIQUE INDEX IF NOT EXISTS idx_deep_analysis_signal_unique ON stock_deep_analysis(signal_id);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_deep_analysis_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_deep_analysis_updated_at ON stock_deep_analysis;
CREATE TRIGGER trigger_deep_analysis_updated_at
    BEFORE UPDATE ON stock_deep_analysis
    FOR EACH ROW
    EXECUTE FUNCTION update_deep_analysis_updated_at();

-- Comments for documentation
COMMENT ON TABLE stock_deep_analysis IS 'Deep Analysis Quality (DAQ) scores for stock scanner signals';
COMMENT ON COLUMN stock_deep_analysis.daq_score IS 'Composite deep analysis score (0-100)';
COMMENT ON COLUMN stock_deep_analysis.daq_grade IS 'Grade based on DAQ score: A+ (85-100), A (70-84), B (60-69), C (50-59), D (<50)';
COMMENT ON COLUMN stock_deep_analysis.mtf_score IS 'Multi-timeframe confluence score - alignment across 1h/4h/1d';
COMMENT ON COLUMN stock_deep_analysis.volume_score IS 'Volume profile score - accumulation/distribution patterns';
COMMENT ON COLUMN stock_deep_analysis.smc_score IS 'Smart Money Concepts score - BOS, CHoCH, order blocks';
COMMENT ON COLUMN stock_deep_analysis.quality_score IS 'Financial quality score - ROE, margins, debt levels';
COMMENT ON COLUMN stock_deep_analysis.catalyst_score IS 'Catalyst timing score (inverted risk) - earnings dates, events';
COMMENT ON COLUMN stock_deep_analysis.news_score IS 'News sentiment score from Finnhub articles';
COMMENT ON COLUMN stock_deep_analysis.regime_score IS 'Market regime alignment score - signal vs SPY trend';
COMMENT ON COLUMN stock_deep_analysis.sector_score IS 'Sector rotation score - relative strength vs market';
