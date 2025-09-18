-- Ichimoku RAG Enhancement Optimization Database Schema
-- Extends the existing optimization system with RAG-enhanced parameter storage

-- Table 1: RAG Enhancement Runs
-- Tracks RAG-enhanced optimization sessions
CREATE TABLE IF NOT EXISTS ichimoku_rag_optimization_runs (
    id SERIAL PRIMARY KEY,
    base_run_id INTEGER REFERENCES ichimoku_optimization_runs(id),
    run_name VARCHAR(100) NOT NULL,
    description TEXT,
    rag_system_version VARCHAR(50),
    tradingview_scripts_count INTEGER DEFAULT 0,
    rag_indicators_count INTEGER DEFAULT 0,
    market_intelligence_enabled BOOLEAN DEFAULT TRUE,
    confluence_scoring_enabled BOOLEAN DEFAULT TRUE,
    mtf_rag_validation_enabled BOOLEAN DEFAULT TRUE,
    start_time TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITHOUT TIME ZONE,
    total_combinations INTEGER,
    status VARCHAR(20) DEFAULT 'running'
);

-- Table 2: RAG Enhancement Configuration
-- Stores RAG-specific configuration parameters for optimization runs
CREATE TABLE IF NOT EXISTS ichimoku_rag_configurations (
    id SERIAL PRIMARY KEY,
    rag_run_id INTEGER REFERENCES ichimoku_rag_optimization_runs(id),

    -- RAG System Configuration
    rag_enabled BOOLEAN DEFAULT TRUE,
    rag_confidence_weight NUMERIC(4,3) DEFAULT 0.20,
    tradingview_integration_enabled BOOLEAN DEFAULT TRUE,
    tradingview_confidence_weight NUMERIC(4,3) DEFAULT 0.15,

    -- Market Intelligence Configuration
    market_intelligence_enabled BOOLEAN DEFAULT TRUE,
    market_regime_adaptation_enabled BOOLEAN DEFAULT TRUE,
    session_adaptation_enabled BOOLEAN DEFAULT TRUE,
    volatility_adaptation_enabled BOOLEAN DEFAULT TRUE,
    adaptive_confidence_threshold BOOLEAN DEFAULT TRUE,

    -- Confluence Scoring Configuration
    confluence_scoring_enabled BOOLEAN DEFAULT TRUE,
    confluence_minimum_indicators INTEGER DEFAULT 3,
    confluence_weight_momentum NUMERIC(4,3) DEFAULT 0.25,
    confluence_weight_trend NUMERIC(4,3) DEFAULT 0.30,
    confluence_weight_volume NUMERIC(4,3) DEFAULT 0.15,
    confluence_weight_support_resistance NUMERIC(4,3) DEFAULT 0.20,
    confluence_weight_pattern NUMERIC(4,3) DEFAULT 0.10,

    -- Multi-Timeframe RAG Validation Configuration
    mtf_rag_validation_enabled BOOLEAN DEFAULT TRUE,
    mtf_rag_timeframe_combinations TEXT, -- JSON array of timeframe combinations
    mtf_rag_consensus_threshold NUMERIC(4,3) DEFAULT 0.60,
    mtf_higher_tf_weight NUMERIC(4,3) DEFAULT 0.40,
    mtf_lower_tf_weight NUMERIC(4,3) DEFAULT 0.25,
    mtf_template_consensus_weight NUMERIC(4,3) DEFAULT 0.35,

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table 3: RAG Enhancement Results
-- Stores results with RAG enhancement metrics
CREATE TABLE IF NOT EXISTS ichimoku_rag_optimization_results (
    id SERIAL PRIMARY KEY,
    rag_run_id INTEGER REFERENCES ichimoku_rag_optimization_runs(id),
    base_result_id INTEGER REFERENCES ichimoku_optimization_results(id),
    epic VARCHAR(50) NOT NULL,

    -- Link to base configuration
    rag_config_id INTEGER REFERENCES ichimoku_rag_configurations(id),

    -- Core Ichimoku Parameters (inherited from base)
    tenkan_period INTEGER NOT NULL DEFAULT 9,
    kijun_period INTEGER NOT NULL DEFAULT 26,
    senkou_b_period INTEGER NOT NULL DEFAULT 52,
    chikou_shift INTEGER NOT NULL DEFAULT 26,
    cloud_shift INTEGER NOT NULL DEFAULT 26,
    confidence_threshold NUMERIC(4,3) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,

    -- RAG Enhancement Metrics
    rag_enhancement_applied BOOLEAN DEFAULT FALSE,
    rag_confidence_boost NUMERIC(6,4) DEFAULT 0.0000,
    rag_pattern_match_count INTEGER DEFAULT 0,
    rag_pattern_strength_avg NUMERIC(5,4) DEFAULT 0.0000,
    rag_template_matches INTEGER DEFAULT 0,
    rag_template_consensus_score NUMERIC(5,4) DEFAULT 0.0000,

    -- TradingView Integration Metrics
    tradingview_integration_applied BOOLEAN DEFAULT FALSE,
    tradingview_technique_type VARCHAR(50),
    tradingview_script_matches INTEGER DEFAULT 0,
    tradingview_confidence_adjustment NUMERIC(6,4) DEFAULT 0.0000,
    tradingview_parameter_adaptations TEXT, -- JSON object of parameter changes

    -- Market Intelligence Metrics
    market_intelligence_applied BOOLEAN DEFAULT FALSE,
    detected_market_regime VARCHAR(20),
    regime_confidence NUMERIC(5,4) DEFAULT 0.0000,
    session_detected VARCHAR(20),
    volatility_level VARCHAR(10),
    adaptive_confidence_applied NUMERIC(6,4) DEFAULT 0.0000,
    regime_parameter_adjustments TEXT, -- JSON object of adjustments

    -- Confluence Scoring Metrics
    confluence_scoring_applied BOOLEAN DEFAULT FALSE,
    confluence_total_score NUMERIC(5,4) DEFAULT 0.0000,
    confluence_level VARCHAR(15), -- LOW, MEDIUM, HIGH, VERY_HIGH
    confluence_bull_score NUMERIC(5,4) DEFAULT 0.0000,
    confluence_bear_score NUMERIC(5,4) DEFAULT 0.0000,
    confluence_indicator_count INTEGER DEFAULT 0,
    confluence_high_confidence_indicators INTEGER DEFAULT 0,
    confluence_weighted_strength NUMERIC(5,4) DEFAULT 0.0000,
    confluence_regime_adjustment NUMERIC(6,4) DEFAULT 0.0000,
    confluence_session_adjustment NUMERIC(6,4) DEFAULT 0.0000,
    confluence_confidence_adjustment NUMERIC(6,4) DEFAULT 0.0000,

    -- Multi-Timeframe RAG Validation Metrics
    mtf_rag_validation_applied BOOLEAN DEFAULT FALSE,
    mtf_validation_passed BOOLEAN DEFAULT FALSE,
    mtf_overall_bias VARCHAR(10),
    mtf_confidence_score NUMERIC(5,4) DEFAULT 0.0000,
    mtf_timeframe_agreement_score NUMERIC(5,4) DEFAULT 0.0000,
    mtf_template_consensus NUMERIC(5,4) DEFAULT 0.0000,
    mtf_higher_tf_support BOOLEAN DEFAULT FALSE,
    mtf_lower_tf_confirmation BOOLEAN DEFAULT FALSE,
    mtf_conflicting_timeframes TEXT, -- JSON array
    mtf_supporting_timeframes TEXT, -- JSON array
    mtf_confidence_adjustment NUMERIC(6,4) DEFAULT 0.0000,

    -- Enhanced Performance Metrics
    rag_enhanced_confidence NUMERIC(5,4) DEFAULT 0.0000,
    total_confidence_adjustment NUMERIC(6,4) DEFAULT 0.0000,
    confidence_improvement_ratio NUMERIC(6,4) DEFAULT 1.0000,

    -- Signal Quality Enhancement Metrics
    enhanced_signal_count INTEGER DEFAULT 0,
    rejected_signal_count INTEGER DEFAULT 0,
    signal_quality_improvement NUMERIC(5,4) DEFAULT 0.0000,
    false_positive_reduction NUMERIC(5,4) DEFAULT 0.0000,

    -- Performance Comparison with Base
    base_win_rate NUMERIC(5,4),
    enhanced_win_rate NUMERIC(5,4),
    win_rate_improvement NUMERIC(6,4) DEFAULT 0.0000,

    base_profit_factor NUMERIC(8,4),
    enhanced_profit_factor NUMERIC(8,4),
    profit_factor_improvement NUMERIC(6,4) DEFAULT 0.0000,

    base_net_pips NUMERIC(10,2),
    enhanced_net_pips NUMERIC(10,2),
    net_pips_improvement NUMERIC(10,2) DEFAULT 0.00,

    base_composite_score NUMERIC(10,6),
    enhanced_composite_score NUMERIC(10,6),
    composite_score_improvement NUMERIC(6,4) DEFAULT 0.0000,

    -- RAG Enhancement Effectiveness Score
    rag_effectiveness_score NUMERIC(8,6), -- Overall effectiveness of RAG enhancements

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table 4: RAG Best Parameters
-- Stores the optimal RAG-enhanced parameter sets
CREATE TABLE IF NOT EXISTS ichimoku_rag_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,

    -- Link to best RAG result
    best_rag_result_id INTEGER REFERENCES ichimoku_rag_optimization_results(id),

    -- Best Core Ichimoku Parameters (RAG-optimized)
    best_tenkan_period INTEGER NOT NULL DEFAULT 9,
    best_kijun_period INTEGER NOT NULL DEFAULT 26,
    best_senkou_b_period INTEGER NOT NULL DEFAULT 52,
    best_chikou_shift INTEGER NOT NULL DEFAULT 26,
    best_cloud_shift INTEGER NOT NULL DEFAULT 26,
    best_confidence_threshold NUMERIC(4,3) NOT NULL,
    best_timeframe VARCHAR(5) NOT NULL,

    -- Best RAG Configuration
    best_rag_enabled BOOLEAN DEFAULT TRUE,
    best_rag_confidence_weight NUMERIC(4,3) DEFAULT 0.20,
    best_tradingview_integration BOOLEAN DEFAULT TRUE,
    best_market_intelligence_enabled BOOLEAN DEFAULT TRUE,
    best_confluence_scoring_enabled BOOLEAN DEFAULT TRUE,
    best_mtf_rag_validation_enabled BOOLEAN DEFAULT TRUE,

    -- Best RAG Enhancement Results
    best_rag_technique_type VARCHAR(50),
    best_detected_regime VARCHAR(20),
    best_confluence_level VARCHAR(15),
    best_mtf_validation_config TEXT, -- JSON object

    -- Best Enhanced Performance Metrics
    best_enhanced_win_rate NUMERIC(5,4),
    best_enhanced_profit_factor NUMERIC(8,4),
    best_enhanced_net_pips NUMERIC(10,2),
    best_enhanced_composite_score NUMERIC(10,6),
    best_rag_effectiveness_score NUMERIC(8,6),

    -- Performance Improvements Over Base Strategy
    win_rate_improvement NUMERIC(6,4),
    profit_factor_improvement NUMERIC(6,4),
    net_pips_improvement NUMERIC(10,2),
    composite_score_improvement NUMERIC(6,4),

    -- RAG Enhancement Details
    confidence_boost_applied NUMERIC(6,4),
    signal_quality_improvement NUMERIC(5,4),
    false_positive_reduction NUMERIC(5,4),

    -- Market Context for RAG Optimization
    optimal_market_regime VARCHAR(20),
    optimal_session VARCHAR(20),
    optimal_volatility_level VARCHAR(10),
    optimal_trading_style VARCHAR(20) DEFAULT 'day_trading',

    -- Maintenance Information
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    optimization_run_date DATE DEFAULT CURRENT_DATE,
    rag_system_version VARCHAR(50),
    next_optimization_due DATE
);

-- Table 5: RAG Enhancement Analytics
-- Tracks the effectiveness of different RAG components
CREATE TABLE IF NOT EXISTS ichimoku_rag_analytics (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL,
    analysis_date DATE DEFAULT CURRENT_DATE,

    -- RAG Component Effectiveness
    rag_enhancer_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    tradingview_parser_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    market_intelligence_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    confluence_scorer_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    mtf_validator_effectiveness NUMERIC(5,4) DEFAULT 0.0000,

    -- Usage Statistics
    total_signals_analyzed INTEGER DEFAULT 0,
    rag_enhanced_signals INTEGER DEFAULT 0,
    enhancement_success_rate NUMERIC(5,4) DEFAULT 0.0000,
    average_confidence_boost NUMERIC(6,4) DEFAULT 0.0000,

    -- Market Condition Analysis
    trending_market_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    ranging_market_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    breakout_market_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    high_volatility_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    low_volatility_effectiveness NUMERIC(5,4) DEFAULT 0.0000,

    -- Session-based Effectiveness
    asian_session_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    london_session_effectiveness NUMERIC(5,4) DEFAULT 0.0000,
    new_york_session_effectiveness NUMERIC(5,4) DEFAULT 0.0000,

    -- Template and Pattern Analytics
    most_effective_tradingview_technique VARCHAR(100),
    most_effective_confluence_pattern VARCHAR(100),
    most_effective_mtf_combination TEXT, -- JSON array

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_rag_results_epic ON ichimoku_rag_optimization_results(epic);
CREATE INDEX IF NOT EXISTS idx_rag_results_effectiveness ON ichimoku_rag_optimization_results(rag_effectiveness_score DESC);
CREATE INDEX IF NOT EXISTS idx_rag_results_run_id ON ichimoku_rag_optimization_results(rag_run_id);
CREATE INDEX IF NOT EXISTS idx_rag_results_created_at ON ichimoku_rag_optimization_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_results_enhanced_composite ON ichimoku_rag_optimization_results(enhanced_composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_rag_results_improvement ON ichimoku_rag_optimization_results(composite_score_improvement DESC);

-- RAG-specific indexes
CREATE INDEX IF NOT EXISTS idx_rag_results_rag_applied ON ichimoku_rag_optimization_results(rag_enhancement_applied);
CREATE INDEX IF NOT EXISTS idx_rag_results_market_intelligence ON ichimoku_rag_optimization_results(market_intelligence_applied);
CREATE INDEX IF NOT EXISTS idx_rag_results_confluence ON ichimoku_rag_optimization_results(confluence_scoring_applied);
CREATE INDEX IF NOT EXISTS idx_rag_results_mtf ON ichimoku_rag_optimization_results(mtf_rag_validation_applied);
CREATE INDEX IF NOT EXISTS idx_rag_results_regime ON ichimoku_rag_optimization_results(detected_market_regime);
CREATE INDEX IF NOT EXISTS idx_rag_results_technique ON ichimoku_rag_optimization_results(tradingview_technique_type);

-- Performance comparison indexes
CREATE INDEX IF NOT EXISTS idx_rag_results_win_rate_improvement ON ichimoku_rag_optimization_results(win_rate_improvement DESC);
CREATE INDEX IF NOT EXISTS idx_rag_results_profit_improvement ON ichimoku_rag_optimization_results(profit_factor_improvement DESC);
CREATE INDEX IF NOT EXISTS idx_rag_results_pips_improvement ON ichimoku_rag_optimization_results(net_pips_improvement DESC);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_rag_analytics_epic_date ON ichimoku_rag_analytics(epic, analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_rag_analytics_effectiveness ON ichimoku_rag_analytics(rag_enhancer_effectiveness DESC);
CREATE INDEX IF NOT EXISTS idx_rag_analytics_success_rate ON ichimoku_rag_analytics(enhancement_success_rate DESC);

-- Configuration indexes
CREATE INDEX IF NOT EXISTS idx_rag_configs_run ON ichimoku_rag_configurations(rag_run_id);
CREATE INDEX IF NOT EXISTS idx_rag_configs_created ON ichimoku_rag_configurations(created_at DESC);

-- Add table comments for documentation
COMMENT ON TABLE ichimoku_rag_optimization_runs IS 'Tracks RAG-enhanced Ichimoku optimization sessions with system version info';
COMMENT ON TABLE ichimoku_rag_configurations IS 'Stores RAG system configuration parameters for optimization runs';
COMMENT ON TABLE ichimoku_rag_optimization_results IS 'Comprehensive results of RAG-enhanced Ichimoku optimizations with detailed metrics';
COMMENT ON TABLE ichimoku_rag_best_parameters IS 'Optimal RAG-enhanced parameter sets for each epic with performance improvements';
COMMENT ON TABLE ichimoku_rag_analytics IS 'Analytics and effectiveness tracking for RAG enhancement components';

-- Add column comments for key metrics
COMMENT ON COLUMN ichimoku_rag_optimization_results.rag_effectiveness_score IS 'Overall effectiveness score of RAG enhancements (0-1 scale)';
COMMENT ON COLUMN ichimoku_rag_optimization_results.confidence_improvement_ratio IS 'Ratio of enhanced confidence to base confidence';
COMMENT ON COLUMN ichimoku_rag_optimization_results.total_confidence_adjustment IS 'Total confidence adjustment from all RAG components';
COMMENT ON COLUMN ichimoku_rag_optimization_results.confluence_level IS 'Confluence analysis level: LOW, MEDIUM, HIGH, VERY_HIGH';
COMMENT ON COLUMN ichimoku_rag_optimization_results.mtf_validation_passed IS 'Whether multi-timeframe RAG validation passed';
COMMENT ON COLUMN ichimoku_rag_optimization_results.tradingview_technique_type IS 'Type of TradingView technique applied (classic, fast, scalping, hybrid)';
COMMENT ON COLUMN ichimoku_rag_optimization_results.detected_market_regime IS 'Market regime detected by intelligence system';

-- Create views for easier data analysis
CREATE OR REPLACE VIEW rag_enhancement_summary AS
SELECT
    epic,
    COUNT(*) as total_tests,
    AVG(rag_effectiveness_score) as avg_effectiveness,
    AVG(composite_score_improvement) as avg_improvement,
    AVG(win_rate_improvement) as avg_win_rate_boost,
    AVG(confidence_improvement_ratio) as avg_confidence_boost,
    SUM(CASE WHEN rag_enhancement_applied THEN 1 ELSE 0 END) as rag_applied_count,
    SUM(CASE WHEN mtf_validation_passed THEN 1 ELSE 0 END) as mtf_passed_count
FROM ichimoku_rag_optimization_results
GROUP BY epic;

CREATE OR REPLACE VIEW best_rag_configurations AS
SELECT
    r.epic,
    r.rag_effectiveness_score,
    r.enhanced_composite_score,
    r.composite_score_improvement,
    r.tradingview_technique_type,
    r.detected_market_regime,
    r.confluence_level,
    c.rag_confidence_weight,
    c.confluence_minimum_indicators,
    c.mtf_rag_consensus_threshold
FROM ichimoku_rag_optimization_results r
JOIN ichimoku_rag_configurations c ON r.rag_config_id = c.id
WHERE r.rag_effectiveness_score = (
    SELECT MAX(rag_effectiveness_score)
    FROM ichimoku_rag_optimization_results r2
    WHERE r2.epic = r.epic
);

-- Display table creation confirmation
SELECT 'Ichimoku RAG optimization tables created successfully!' as status;