-- EMA Strategy Parameter Optimization Database Schema
-- Run this to create optimization tracking tables in PostgreSQL forex database

-- =============================================================================
-- OPTIMIZATION RUNS TRACKING
-- =============================================================================

-- Track optimization runs and sessions
CREATE TABLE IF NOT EXISTS ema_optimization_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(100) NOT NULL,
    description TEXT,
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    total_combinations INTEGER,
    completed_combinations INTEGER DEFAULT 0,
    epics_tested TEXT[], -- Array of epics included in this run
    status VARCHAR(20) DEFAULT 'running', -- running, completed, failed, cancelled
    created_by VARCHAR(50) DEFAULT 'system',
    
    -- Run configuration
    backtest_days INTEGER DEFAULT 30,
    min_signals_threshold INTEGER DEFAULT 20,
    
    CONSTRAINT valid_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- Index for run queries
CREATE INDEX IF NOT EXISTS idx_optimization_runs_status ON ema_optimization_runs(status, start_time DESC);

-- =============================================================================
-- DETAILED OPTIMIZATION RESULTS
-- =============================================================================

-- Store individual parameter test results with comprehensive metrics
CREATE TABLE IF NOT EXISTS ema_optimization_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES ema_optimization_runs(id) ON DELETE CASCADE,
    
    -- Trading Parameters
    epic VARCHAR(50) NOT NULL,
    ema_config VARCHAR(20) NOT NULL,
    confidence_threshold DECIMAL(4,3) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    smart_money_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Risk Management Parameters
    stop_loss_pips DECIMAL(6,1) NOT NULL,
    take_profit_pips DECIMAL(6,1) NOT NULL,
    risk_reward_ratio DECIMAL(6,3) NOT NULL,
    breakeven_trigger_pips DECIMAL(6,1),
    profit_protection_pips DECIMAL(6,1),
    trailing_start_pips DECIMAL(6,1),
    trailing_ratio DECIMAL(3,2),
    
    -- Core Performance Metrics
    total_signals INTEGER NOT NULL DEFAULT 0,
    valid_signals INTEGER NOT NULL DEFAULT 0, -- Signals with enough lookback data
    win_rate DECIMAL(5,4), -- Percentage as decimal (0.6534 = 65.34%)
    loss_rate DECIMAL(5,4),
    
    -- Profit/Loss Analysis
    avg_profit_pips DECIMAL(8,2),
    avg_loss_pips DECIMAL(8,2),
    max_profit_pips DECIMAL(8,2),
    max_loss_pips DECIMAL(8,2),
    total_profit_pips DECIMAL(10,2),
    total_loss_pips DECIMAL(10,2),
    net_pips DECIMAL(10,2),
    
    -- Advanced Metrics
    profit_factor DECIMAL(8,4), -- Total profit / Total loss
    expectancy_per_trade DECIMAL(8,3), -- Average pips per trade
    risk_reward_achieved DECIMAL(6,3), -- Actual R:R achieved
    
    -- Risk Metrics
    max_drawdown_pips DECIMAL(8,2),
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    largest_win_pips DECIMAL(8,2),
    largest_loss_pips DECIMAL(8,2),
    
    -- Advanced Performance Indicators
    sharpe_ratio DECIMAL(6,4),
    calmar_ratio DECIMAL(6,4), -- Annual return / Max drawdown
    recovery_factor DECIMAL(6,4), -- Net profit / Max drawdown
    
    -- Signal Distribution
    bull_signals INTEGER DEFAULT 0,
    bear_signals INTEGER DEFAULT 0,
    
    -- Exit Strategy Breakdown
    profit_target_exits INTEGER DEFAULT 0,
    trailing_stop_wins INTEGER DEFAULT 0,
    trailing_stop_losses INTEGER DEFAULT 0,
    breakeven_exits INTEGER DEFAULT 0,
    stop_loss_exits INTEGER DEFAULT 0,
    timeout_exits INTEGER DEFAULT 0,
    
    -- Quality Metrics
    avg_confidence DECIMAL(5,4),
    avg_signal_quality DECIMAL(5,4),
    signal_frequency_per_day DECIMAL(6,3), -- Signals per day average
    
    -- Execution Metadata
    backtest_duration_seconds INTEGER,
    data_range_start TIMESTAMP,
    data_range_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Performance ranking (calculated field)
    composite_score DECIMAL(10,6) -- Calculated: win_rate * profit_factor * (net_pips/100)
);

-- Indexes for fast epic-based queries
CREATE INDEX IF NOT EXISTS idx_optimization_results_epic ON ema_optimization_results(epic);
CREATE INDEX IF NOT EXISTS idx_optimization_results_run_epic ON ema_optimization_results(run_id, epic);
CREATE INDEX IF NOT EXISTS idx_optimization_results_performance ON ema_optimization_results(epic, composite_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_optimization_results_win_rate ON ema_optimization_results(epic, win_rate DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_optimization_results_profit_factor ON ema_optimization_results(epic, profit_factor DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_optimization_results_net_pips ON ema_optimization_results(epic, net_pips DESC NULLS LAST);

-- =============================================================================
-- BEST PARAMETERS SUMMARY
-- =============================================================================

-- Store the best discovered parameters for each epic
CREATE TABLE IF NOT EXISTS ema_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,
    
    -- Optimal Strategy Parameters
    best_ema_config VARCHAR(20) NOT NULL,
    best_confidence_threshold DECIMAL(4,3) NOT NULL,
    best_timeframe VARCHAR(5) NOT NULL,
    best_smart_money_enabled BOOLEAN NOT NULL,
    
    -- Optimal Risk Management
    optimal_stop_loss_pips DECIMAL(6,1) NOT NULL,
    optimal_take_profit_pips DECIMAL(6,1) NOT NULL,
    optimal_risk_reward_ratio DECIMAL(6,3) NOT NULL,
    optimal_trailing_config JSONB, -- Store trailing stop configuration
    
    -- Best Performance Achieved
    best_win_rate DECIMAL(5,4) NOT NULL,
    best_profit_factor DECIMAL(8,4) NOT NULL,
    best_expectancy DECIMAL(8,3) NOT NULL,
    best_net_pips DECIMAL(10,2) NOT NULL,
    best_composite_score DECIMAL(10,6) NOT NULL,
    
    -- Best Result Metadata
    best_result_id INTEGER REFERENCES ema_optimization_results(id),
    total_signals_in_best INTEGER,
    optimization_run_id INTEGER REFERENCES ema_optimization_runs(id),
    
    -- Currency Pair Insights
    pair_characteristics JSONB, -- Store epic-specific insights
    volatility_profile VARCHAR(20), -- low, medium, high
    preferred_sessions TEXT[], -- asian, london, new_york, overlap
    
    -- Update Tracking
    last_updated TIMESTAMP DEFAULT NOW(),
    last_optimization_date TIMESTAMP DEFAULT NOW(),
    optimization_confidence VARCHAR(20) DEFAULT 'high' -- high, medium, low based on sample size
);

-- =============================================================================
-- PARAMETER SENSITIVITY ANALYSIS
-- =============================================================================

-- Track how sensitive each epic is to parameter changes
CREATE TABLE IF NOT EXISTS ema_parameter_sensitivity (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL,
    parameter_name VARCHAR(50) NOT NULL, -- ema_config, confidence_threshold, stop_loss_pips, etc.
    parameter_value TEXT NOT NULL,
    
    -- Performance Impact
    avg_win_rate DECIMAL(5,4),
    avg_profit_factor DECIMAL(8,4),
    avg_composite_score DECIMAL(10,6),
    test_count INTEGER,
    
    -- Statistical Significance
    std_deviation DECIMAL(8,4),
    confidence_interval_95_lower DECIMAL(8,4),
    confidence_interval_95_upper DECIMAL(8,4),
    
    optimization_run_id INTEGER REFERENCES ema_optimization_runs(id),
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_epic_parameter_value UNIQUE(epic, parameter_name, parameter_value, optimization_run_id)
);

-- Index for parameter sensitivity queries
CREATE INDEX IF NOT EXISTS idx_parameter_sensitivity_epic ON ema_parameter_sensitivity(epic, parameter_name);

-- =============================================================================
-- OPTIMIZATION INSIGHTS & RECOMMENDATIONS
-- =============================================================================

-- Store insights and recommendations discovered during optimization
CREATE TABLE IF NOT EXISTS ema_optimization_insights (
    id SERIAL PRIMARY KEY,
    optimization_run_id INTEGER REFERENCES ema_optimization_runs(id),
    epic VARCHAR(50),
    insight_type VARCHAR(50), -- parameter_correlation, risk_preference, session_preference, etc.
    insight_title VARCHAR(200),
    insight_description TEXT,
    supporting_data JSONB,
    confidence_level VARCHAR(20), -- high, medium, low
    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- UTILITY FUNCTIONS
-- =============================================================================

-- Function to calculate composite score
CREATE OR REPLACE FUNCTION calculate_composite_score(
    p_win_rate DECIMAL,
    p_profit_factor DECIMAL, 
    p_net_pips DECIMAL
) RETURNS DECIMAL AS $$
BEGIN
    -- Avoid division by zero and handle nulls
    IF p_win_rate IS NULL OR p_profit_factor IS NULL OR p_net_pips IS NULL THEN
        RETURN 0;
    END IF;
    
    IF p_win_rate = 0 OR p_profit_factor = 0 THEN
        RETURN 0;
    END IF;
    
    -- Composite score: win_rate * profit_factor * (net_pips/100)
    -- This gives higher scores to strategies with good win rates, profit factors, and absolute returns
    RETURN p_win_rate * p_profit_factor * (p_net_pips / 100.0);
END;
$$ LANGUAGE plpgsql;

-- Function to update composite scores
CREATE OR REPLACE FUNCTION update_composite_scores(run_id_param INTEGER DEFAULT NULL) 
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    UPDATE ema_optimization_results 
    SET composite_score = calculate_composite_score(win_rate, profit_factor, net_pips)
    WHERE (run_id_param IS NULL OR run_id = run_id_param);
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SAMPLE DATA VIEWS
-- =============================================================================

-- View for top performing configurations per epic
CREATE OR REPLACE VIEW v_top_epic_configurations AS
SELECT 
    epic,
    ema_config,
    confidence_threshold,
    timeframe,
    smart_money_enabled,
    stop_loss_pips,
    take_profit_pips,
    risk_reward_ratio,
    win_rate,
    profit_factor,
    net_pips,
    composite_score,
    total_signals,
    ROW_NUMBER() OVER (PARTITION BY epic ORDER BY composite_score DESC NULLS LAST) as rank
FROM ema_optimization_results
WHERE total_signals >= 20  -- Minimum signal threshold
AND composite_score IS NOT NULL;

-- View for parameter performance summary
CREATE OR REPLACE VIEW v_parameter_performance_summary AS
SELECT 
    epic,
    ema_config,
    COUNT(*) as test_count,
    AVG(win_rate) as avg_win_rate,
    AVG(profit_factor) as avg_profit_factor,
    AVG(net_pips) as avg_net_pips,
    AVG(composite_score) as avg_composite_score,
    MAX(composite_score) as best_composite_score
FROM ema_optimization_results
WHERE total_signals >= 20
GROUP BY epic, ema_config
ORDER BY epic, avg_composite_score DESC NULLS LAST;

-- =============================================================================
-- INITIAL DATA SETUP
-- =============================================================================

-- Insert sample optimization run for testing
INSERT INTO ema_optimization_runs (run_name, description, backtest_days, min_signals_threshold, status)
VALUES 
    ('initial_setup', 'Initial setup and schema validation', 30, 20, 'completed')
ON CONFLICT DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ EMA Optimization schema created successfully!';
    RAISE NOTICE '📊 Tables created: ema_optimization_runs, ema_optimization_results, ema_best_parameters, ema_parameter_sensitivity, ema_optimization_insights';
    RAISE NOTICE '🔍 Views created: v_top_epic_configurations, v_parameter_performance_summary';
    RAISE NOTICE '⚡ Functions created: calculate_composite_score(), update_composite_scores()';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 Ready to run parameter optimization!';
END $$;