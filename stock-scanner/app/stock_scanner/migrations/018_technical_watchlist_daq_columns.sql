-- Migration: 018_technical_watchlist_daq_columns.sql
-- Description: Add DAQ (Deep Analysis Quality) columns to stock_watchlist_results table
-- Date: 2026-01-12

-- Add DAQ score columns to technical watchlist results
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_score INTEGER CHECK (daq_score >= 0 AND daq_score <= 100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_grade VARCHAR(2) CHECK (daq_grade IN ('A+', 'A', 'B', 'C', 'D'));

-- Technical component scores
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_mtf_score INTEGER CHECK (daq_mtf_score >= 0 AND daq_mtf_score <= 100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_volume_score INTEGER CHECK (daq_volume_score >= 0 AND daq_volume_score <= 100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_smc_score INTEGER CHECK (daq_smc_score >= 0 AND daq_smc_score <= 100);

-- Fundamental component scores
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_quality_score INTEGER CHECK (daq_quality_score >= 0 AND daq_quality_score <= 100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_catalyst_score INTEGER CHECK (daq_catalyst_score >= 0 AND daq_catalyst_score <= 100);

-- Contextual component scores
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_news_score INTEGER CHECK (daq_news_score >= 0 AND daq_news_score <= 100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_regime_score INTEGER CHECK (daq_regime_score >= 0 AND daq_regime_score <= 100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_sector_score INTEGER CHECK (daq_sector_score >= 0 AND daq_sector_score <= 100);

-- Risk flags
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_earnings_risk BOOLEAN DEFAULT FALSE;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_high_short_interest BOOLEAN DEFAULT FALSE;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_sector_underperforming BOOLEAN DEFAULT FALSE;

-- Metadata
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS daq_analyzed_at TIMESTAMPTZ;

-- Index for DAQ queries
CREATE INDEX IF NOT EXISTS idx_watchlist_results_daq_score ON stock_watchlist_results(daq_score DESC) WHERE daq_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_watchlist_results_daq_grade ON stock_watchlist_results(daq_grade) WHERE daq_grade IS NOT NULL;

-- Comments
COMMENT ON COLUMN stock_watchlist_results.daq_score IS 'Deep Analysis Quality composite score (0-100)';
COMMENT ON COLUMN stock_watchlist_results.daq_grade IS 'DAQ grade: A+ (85-100), A (70-84), B (60-69), C (50-59), D (<50)';
COMMENT ON COLUMN stock_watchlist_results.daq_mtf_score IS 'Multi-timeframe confluence score';
COMMENT ON COLUMN stock_watchlist_results.daq_volume_score IS 'Volume analysis score';
COMMENT ON COLUMN stock_watchlist_results.daq_smc_score IS 'Smart Money Concepts score';
COMMENT ON COLUMN stock_watchlist_results.daq_quality_score IS 'Financial quality score';
COMMENT ON COLUMN stock_watchlist_results.daq_catalyst_score IS 'Catalyst timing score (inverted risk)';
COMMENT ON COLUMN stock_watchlist_results.daq_news_score IS 'News sentiment score';
COMMENT ON COLUMN stock_watchlist_results.daq_regime_score IS 'Market regime alignment score';
COMMENT ON COLUMN stock_watchlist_results.daq_sector_score IS 'Sector rotation score';
