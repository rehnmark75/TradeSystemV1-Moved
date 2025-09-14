-- TradingView Scripts PostgreSQL Schema
-- Creates tables and indexes for TradingView script storage and search

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tradingview schema
CREATE SCHEMA IF NOT EXISTS tradingview;

-- Main scripts table
CREATE TABLE IF NOT EXISTS tradingview.scripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(255) NOT NULL,
    description TEXT,
    code TEXT,
    open_source BOOLEAN DEFAULT TRUE,
    likes INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_url TEXT,
    script_type VARCHAR(50) DEFAULT 'strategy', -- 'strategy' or 'indicator'
    strategy_type VARCHAR(50), -- 'trending', 'momentum', 'scalping', etc.
    indicators TEXT[], -- Array of indicator names
    signals TEXT[], -- Array of signal types
    timeframes TEXT[], -- Array of supported timeframes
    parameters JSONB, -- Extracted parameters
    metadata JSONB -- Additional metadata
);

-- Script analysis table (for processed analysis results)
CREATE TABLE IF NOT EXISTS tradingview.script_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    script_id UUID NOT NULL REFERENCES tradingview.scripts(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL, -- 'pine_extraction', 'signal_analysis', etc.
    analysis_data JSONB NOT NULL,
    complexity_score DECIMAL(3,2),
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Script imports tracking table
CREATE TABLE IF NOT EXISTS tradingview.script_imports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    script_id UUID NOT NULL REFERENCES tradingview.scripts(id) ON DELETE CASCADE,
    target_config VARCHAR(100) NOT NULL, -- e.g., 'ema_strategy', 'macd_strategy'
    preset_name VARCHAR(100) NOT NULL,
    import_status VARCHAR(20) DEFAULT 'active', -- 'active', 'removed'
    imported_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    imported_by VARCHAR(100) DEFAULT 'system',
    config_path TEXT,
    performance_data JSONB -- Backtest results, etc.
);

-- Script performance tracking
CREATE TABLE IF NOT EXISTS tradingview.script_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    script_id UUID NOT NULL REFERENCES tradingview.scripts(id) ON DELETE CASCADE,
    test_type VARCHAR(50) NOT NULL, -- 'backtest', 'forward_test', 'optimization'
    test_period_start DATE,
    test_period_end DATE,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    performance_metrics JSONB, -- Win rate, profit factor, etc.
    parameters_used JSONB,
    tested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_scripts_slug ON tradingview.scripts(slug);
CREATE INDEX IF NOT EXISTS idx_scripts_strategy_type ON tradingview.scripts(strategy_type);
CREATE INDEX IF NOT EXISTS idx_scripts_script_type ON tradingview.scripts(script_type);
CREATE INDEX IF NOT EXISTS idx_scripts_likes ON tradingview.scripts(likes DESC);
CREATE INDEX IF NOT EXISTS idx_scripts_views ON tradingview.scripts(views DESC);
CREATE INDEX IF NOT EXISTS idx_scripts_created_at ON tradingview.scripts(created_at DESC);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_scripts_title_fts ON tradingview.scripts USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_scripts_description_fts ON tradingview.scripts USING gin(to_tsvector('english', description));
CREATE INDEX IF NOT EXISTS idx_scripts_code_fts ON tradingview.scripts USING gin(to_tsvector('english', code));
CREATE INDEX IF NOT EXISTS idx_scripts_author_fts ON tradingview.scripts USING gin(to_tsvector('english', author));

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_scripts_type_likes ON tradingview.scripts(strategy_type, likes DESC);
CREATE INDEX IF NOT EXISTS idx_scripts_indicators_gin ON tradingview.scripts USING gin(indicators);
CREATE INDEX IF NOT EXISTS idx_scripts_signals_gin ON tradingview.scripts USING gin(signals);

-- Analysis table indexes
CREATE INDEX IF NOT EXISTS idx_analysis_script_id ON tradingview.script_analysis(script_id);
CREATE INDEX IF NOT EXISTS idx_analysis_type ON tradingview.script_analysis(analysis_type);

-- Imports table indexes
CREATE INDEX IF NOT EXISTS idx_imports_script_id ON tradingview.script_imports(script_id);
CREATE INDEX IF NOT EXISTS idx_imports_target_config ON tradingview.script_imports(target_config);
CREATE INDEX IF NOT EXISTS idx_imports_status ON tradingview.script_imports(import_status);

-- Performance table indexes
CREATE INDEX IF NOT EXISTS idx_performance_script_id ON tradingview.script_performance(script_id);
CREATE INDEX IF NOT EXISTS idx_performance_test_type ON tradingview.script_performance(test_type);
CREATE INDEX IF NOT EXISTS idx_performance_symbol ON tradingview.script_performance(symbol);

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION tradingview.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_scripts_updated_at 
    BEFORE UPDATE ON tradingview.scripts 
    FOR EACH ROW EXECUTE FUNCTION tradingview.update_updated_at_column();

-- Views for common queries
CREATE OR REPLACE VIEW tradingview.popular_scripts AS
SELECT 
    s.*,
    (s.likes::float / GREATEST(s.views::float, 1)) * 100 as engagement_rate
FROM tradingview.scripts s
WHERE s.open_source = true
ORDER BY s.likes DESC, s.views DESC;

CREATE OR REPLACE VIEW tradingview.script_summary AS
SELECT 
    strategy_type,
    script_type,
    COUNT(*) as script_count,
    AVG(likes) as avg_likes,
    AVG(views) as avg_views,
    MAX(likes) as max_likes,
    MAX(views) as max_views
FROM tradingview.scripts 
GROUP BY strategy_type, script_type;

-- Comments
COMMENT ON SCHEMA tradingview IS 'TradingView scripts integration schema';
COMMENT ON TABLE tradingview.scripts IS 'TradingView scripts with metadata and code';
COMMENT ON TABLE tradingview.script_analysis IS 'Analysis results for scripts';
COMMENT ON TABLE tradingview.script_imports IS 'Tracking of imported scripts into TradeSystemV1';
COMMENT ON TABLE tradingview.script_performance IS 'Performance testing results for scripts';
COMMENT ON VIEW tradingview.popular_scripts IS 'Scripts ordered by popularity metrics';
COMMENT ON VIEW tradingview.script_summary IS 'Aggregate statistics by script category';