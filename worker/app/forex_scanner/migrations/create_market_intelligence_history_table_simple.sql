-- Market Intelligence History Table - Simplified Version
-- Creates a dedicated table for storing market intelligence data from each scan cycle
-- Allows analysis of market conditions independent of signal generation
-- Created: 2025-09-22

-- Drop table if exists (for testing/development)
-- DROP TABLE IF EXISTS market_intelligence_history;

-- Main market intelligence history table
CREATE TABLE IF NOT EXISTS market_intelligence_history (
    id SERIAL PRIMARY KEY,

    -- Scan metadata
    scan_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    scan_cycle_id VARCHAR(64), -- Optional identifier for scan cycles
    epic_list TEXT[], -- Array of epics analyzed in this scan
    epic_count INTEGER DEFAULT 0,

    -- Market regime analysis
    dominant_regime VARCHAR(30) NOT NULL, -- trending, ranging, breakout, reversal, high_volatility, low_volatility
    regime_confidence DECIMAL(5,4) NOT NULL DEFAULT 0.5,
    regime_scores JSON, -- Full regime scores object

    -- Session analysis
    current_session VARCHAR(20) NOT NULL, -- asian, london, new_york, overlap
    session_volatility VARCHAR(20), -- low, medium, high, very_high
    session_characteristics TEXT[], -- Array of session characteristics
    optimal_timeframes TEXT[], -- Recommended timeframes for this session

    -- Market context and strength
    market_bias VARCHAR(20), -- bullish, bearish, neutral
    average_trend_strength DECIMAL(6,4),
    average_volatility DECIMAL(6,4),
    directional_consensus DECIMAL(6,4),
    market_efficiency DECIMAL(6,4),
    volatility_percentile DECIMAL(5,2),

    -- Correlation and currency analysis
    correlation_analysis JSON, -- Full correlation analysis object
    currency_strength JSON, -- Currency strength rankings
    risk_sentiment VARCHAR(20), -- risk_on, risk_off, neutral

    -- Strategy recommendations
    recommended_strategy VARCHAR(30),
    confidence_threshold DECIMAL(5,4),
    position_sizing_recommendation VARCHAR(20), -- REDUCED, NORMAL, INCREASED
    strategy_adjustments TEXT,

    -- Technical analysis summary
    market_strength_summary JSON, -- Detailed market strength metrics
    pair_analyses JSON, -- Individual pair analysis results
    support_resistance_levels JSON, -- Key levels identified

    -- Intelligence metadata
    intelligence_source VARCHAR(50) DEFAULT 'MarketIntelligenceEngine',
    analysis_duration_ms INTEGER, -- Time taken to generate intelligence
    data_quality_score DECIMAL(5,4), -- Quality of underlying data
    successful_pair_analyses INTEGER DEFAULT 0,
    failed_pair_analyses INTEGER DEFAULT 0,

    -- Indexable fields for fast queries (duplicated from JSON for performance)
    regime_trending_score DECIMAL(5,4),
    regime_ranging_score DECIMAL(5,4),
    regime_breakout_score DECIMAL(5,4),
    regime_reversal_score DECIMAL(5,4),
    regime_high_vol_score DECIMAL(5,4),
    regime_low_vol_score DECIMAL(5,4),

    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_market_intelligence_scan_timestamp ON market_intelligence_history(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_dominant_regime ON market_intelligence_history(dominant_regime);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_regime_confidence ON market_intelligence_history(regime_confidence);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_session ON market_intelligence_history(current_session);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_market_bias ON market_intelligence_history(market_bias);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_volatility ON market_intelligence_history(session_volatility);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_risk_sentiment ON market_intelligence_history(risk_sentiment);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_market_intelligence_regime_session ON market_intelligence_history(dominant_regime, current_session);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_timestamp_regime ON market_intelligence_history(scan_timestamp, dominant_regime);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_confidence_regime ON market_intelligence_history(regime_confidence, dominant_regime);

-- Partial indexes for high-confidence analysis
CREATE INDEX IF NOT EXISTS idx_market_intelligence_high_confidence ON market_intelligence_history(scan_timestamp) WHERE regime_confidence > 0.7;
CREATE INDEX IF NOT EXISTS idx_market_intelligence_trending_high_conf ON market_intelligence_history(scan_timestamp) WHERE dominant_regime = 'trending' AND regime_confidence > 0.8;

-- Basic GIN index for array queries only (no JSON operator class issues)
CREATE INDEX IF NOT EXISTS idx_market_intelligence_epic_list ON market_intelligence_history USING GIN(epic_list);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_market_intelligence_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
DROP TRIGGER IF EXISTS trigger_update_market_intelligence_updated_at ON market_intelligence_history;
CREATE TRIGGER trigger_update_market_intelligence_updated_at
    BEFORE UPDATE ON market_intelligence_history
    FOR EACH ROW EXECUTE FUNCTION update_market_intelligence_updated_at();