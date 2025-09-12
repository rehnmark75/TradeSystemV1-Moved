-- SMC Optimization Database Tables
-- Creates tables for Smart Money Concepts strategy parameter optimization
-- Follows the same pattern as EMA and MACD optimization systems

-- =============================================================================
-- SMC OPTIMIZATION RUNS TABLE
-- Tracks optimization sessions and metadata
-- =============================================================================
CREATE TABLE IF NOT EXISTS smc_optimization_runs (
    run_id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    optimization_mode VARCHAR(20) DEFAULT 'full', -- 'smart_presets', 'fast', 'full'
    days_analyzed INTEGER NOT NULL,
    total_combinations INTEGER,
    completed_combinations INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    best_score DECIMAL(10, 6),
    best_parameters JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_smc_runs_epic_status ON smc_optimization_runs(epic, status);
CREATE INDEX IF NOT EXISTS idx_smc_runs_start_time ON smc_optimization_runs(start_time);

-- =============================================================================
-- SMC OPTIMIZATION RESULTS TABLE  
-- Stores detailed results for each parameter combination tested
-- =============================================================================
CREATE TABLE IF NOT EXISTS smc_optimization_results (
    result_id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES smc_optimization_runs(run_id) ON DELETE CASCADE,
    epic VARCHAR(50) NOT NULL,
    
    -- SMC Configuration Parameters
    smc_config VARCHAR(20) NOT NULL,         -- 'default', 'moderate', 'conservative', etc.
    confidence_level DECIMAL(3, 2) NOT NULL, -- 0.40 to 0.80
    timeframe VARCHAR(5) NOT NULL,           -- '5m', '15m', '1h'
    use_smart_money BOOLEAN DEFAULT true,    -- Enable Smart Money features
    
    -- Market Structure Parameters
    swing_length INTEGER NOT NULL,           -- 2-10: Length for swing detection
    structure_confirmation INTEGER NOT NULL, -- 1-8: Bars needed for structure confirmation
    bos_threshold DECIMAL(8, 5) NOT NULL,   -- Break of Structure threshold
    choch_threshold DECIMAL(8, 5) NOT NULL, -- Change of Character threshold
    
    -- Order Block Parameters
    order_block_length INTEGER NOT NULL,      -- 1-8: Min length for order blocks
    order_block_volume_factor DECIMAL(3, 1) NOT NULL, -- 1.0-2.5: Volume multiplier
    order_block_buffer DECIMAL(3, 1) NOT NULL,        -- 0.5-5.0: Pips buffer around zones
    max_order_blocks INTEGER NOT NULL,                 -- 2-10: Max order blocks to track
    
    -- Fair Value Gap Parameters
    fvg_min_size DECIMAL(3, 1) NOT NULL,     -- 1.0-8.0: Minimum FVG size in pips
    fvg_max_age INTEGER NOT NULL,            -- 10-40: Maximum bars to keep FVG active
    fvg_fill_threshold DECIMAL(3, 2) NOT NULL, -- 0.2-0.8: Fill threshold to consider closed
    
    -- Supply/Demand Zone Parameters
    zone_min_touches INTEGER NOT NULL,       -- 1-4: Min touches to create zone
    zone_max_age INTEGER NOT NULL,           -- 20-100: Max bars to keep zone active
    zone_strength_factor DECIMAL(3, 1) NOT NULL, -- 1.0-2.5: Volume factor for strong zones
    
    -- Signal Generation Parameters
    confluence_required DECIMAL(3, 1) NOT NULL,  -- 1.0-4.0: Min confluence factors
    min_risk_reward DECIMAL(3, 1) NOT NULL,      -- 1.0-3.0: Min R:R ratio
    max_distance_to_zone DECIMAL(4, 1) NOT NULL, -- 3.0-20.0: Max pips to zone
    min_signal_confidence DECIMAL(3, 2) NOT NULL, -- 0.30-0.80: Min signal confidence
    
    -- Multi-timeframe Parameters
    use_higher_tf BOOLEAN DEFAULT true,      -- Use higher timeframe confirmation
    higher_tf_multiplier INTEGER NOT NULL,   -- 2-6: Higher TF multiplier
    mtf_confluence_weight DECIMAL(3, 2) NOT NULL, -- 0.5-1.0: MTF confluence weight
    
    -- Risk Management Parameters
    stop_loss_pips DECIMAL(4, 1) NOT NULL,   -- 5.0-25.0: Stop loss in pips
    take_profit_pips DECIMAL(4, 1) NOT NULL, -- 10.0-50.0: Take profit in pips
    risk_reward_ratio DECIMAL(3, 1) NOT NULL, -- 1.0-3.0: R:R ratio
    
    -- Performance Metrics
    total_signals INTEGER DEFAULT 0,
    winning_signals INTEGER DEFAULT 0,
    losing_signals INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2) DEFAULT 0.00,     -- 0.00-100.00%
    
    -- SMC-Specific Metrics
    structure_breaks_detected INTEGER DEFAULT 0,  -- Total structure breaks found
    order_block_reactions INTEGER DEFAULT 0,      -- Successful order block reactions
    fvg_reactions INTEGER DEFAULT 0,              -- Fair value gap reactions
    liquidity_sweeps INTEGER DEFAULT 0,           -- Liquidity sweep events
    confluence_accuracy DECIMAL(5, 2) DEFAULT 0.00, -- Accuracy of confluence signals
    
    -- Financial Metrics
    total_pips_gained DECIMAL(8, 2) DEFAULT 0.00,
    total_pips_lost DECIMAL(8, 2) DEFAULT 0.00,
    net_pips DECIMAL(8, 2) DEFAULT 0.00,
    average_win_pips DECIMAL(6, 2) DEFAULT 0.00,
    average_loss_pips DECIMAL(6, 2) DEFAULT 0.00,
    profit_factor DECIMAL(6, 3) DEFAULT 0.000,   -- Gross profit / gross loss
    max_drawdown_pips DECIMAL(6, 2) DEFAULT 0.00,
    
    -- Composite Score (win_rate × profit_factor × net_pips/100)
    performance_score DECIMAL(10, 6) DEFAULT 0.000000,
    
    -- Timestamps
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(run_id, epic, smc_config, confidence_level, timeframe, swing_length, 
           structure_confirmation, order_block_length, fvg_min_size, 
           confluence_required, min_risk_reward, stop_loss_pips, take_profit_pips)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_smc_results_epic_score ON smc_optimization_results(epic, performance_score);
CREATE INDEX IF NOT EXISTS idx_smc_results_run_id ON smc_optimization_results(run_id);
CREATE INDEX IF NOT EXISTS idx_smc_results_win_rate ON smc_optimization_results(win_rate);
CREATE INDEX IF NOT EXISTS idx_smc_results_profit_factor ON smc_optimization_results(profit_factor);
CREATE INDEX IF NOT EXISTS idx_smc_results_net_pips ON smc_optimization_results(net_pips);
CREATE INDEX IF NOT EXISTS idx_smc_results_config_timeframe ON smc_optimization_results(smc_config, timeframe);

-- =============================================================================
-- SMC BEST PARAMETERS TABLE
-- Stores the optimal parameters for each epic (updated after each optimization run)
-- =============================================================================
CREATE TABLE IF NOT EXISTS smc_best_parameters (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) UNIQUE NOT NULL,
    
    -- Reference to optimization run
    optimization_run_id INTEGER REFERENCES smc_optimization_runs(run_id),
    result_id INTEGER REFERENCES smc_optimization_results(result_id),
    
    -- Best SMC Configuration
    best_smc_config VARCHAR(20) NOT NULL,
    best_confidence_level DECIMAL(3, 2) NOT NULL,
    best_timeframe VARCHAR(5) NOT NULL,
    use_smart_money BOOLEAN DEFAULT true,
    
    -- Optimal Market Structure Parameters
    optimal_swing_length INTEGER NOT NULL,
    optimal_structure_confirmation INTEGER NOT NULL,
    optimal_bos_threshold DECIMAL(8, 5) NOT NULL,
    optimal_choch_threshold DECIMAL(8, 5) NOT NULL,
    
    -- Optimal Order Block Parameters
    optimal_order_block_length INTEGER NOT NULL,
    optimal_order_block_volume_factor DECIMAL(3, 1) NOT NULL,
    optimal_order_block_buffer DECIMAL(3, 1) NOT NULL,
    optimal_max_order_blocks INTEGER NOT NULL,
    
    -- Optimal Fair Value Gap Parameters
    optimal_fvg_min_size DECIMAL(3, 1) NOT NULL,
    optimal_fvg_max_age INTEGER NOT NULL,
    optimal_fvg_fill_threshold DECIMAL(3, 2) NOT NULL,
    
    -- Optimal Supply/Demand Zone Parameters
    optimal_zone_min_touches INTEGER NOT NULL,
    optimal_zone_max_age INTEGER NOT NULL,
    optimal_zone_strength_factor DECIMAL(3, 1) NOT NULL,
    
    -- Optimal Signal Generation Parameters
    optimal_confluence_required DECIMAL(3, 1) NOT NULL,
    optimal_min_risk_reward DECIMAL(3, 1) NOT NULL,
    optimal_max_distance_to_zone DECIMAL(4, 1) NOT NULL,
    optimal_min_signal_confidence DECIMAL(3, 2) NOT NULL,
    
    -- Optimal Multi-timeframe Parameters
    optimal_use_higher_tf BOOLEAN DEFAULT true,
    optimal_higher_tf_multiplier INTEGER NOT NULL,
    optimal_mtf_confluence_weight DECIMAL(3, 2) NOT NULL,
    
    -- Optimal Risk Management
    optimal_stop_loss_pips DECIMAL(4, 1) NOT NULL,
    optimal_take_profit_pips DECIMAL(4, 1) NOT NULL,
    optimal_risk_reward_ratio DECIMAL(3, 1) NOT NULL,
    
    -- Performance Metrics of Best Configuration
    best_win_rate DECIMAL(5, 2) NOT NULL,
    best_profit_factor DECIMAL(6, 3) NOT NULL,
    best_net_pips DECIMAL(8, 2) NOT NULL,
    best_performance_score DECIMAL(10, 6) NOT NULL,
    
    -- SMC-Specific Performance
    structure_break_accuracy DECIMAL(5, 2) DEFAULT 0.00,
    order_block_success_rate DECIMAL(5, 2) DEFAULT 0.00,
    fvg_success_rate DECIMAL(5, 2) DEFAULT 0.00,
    avg_confluence_score DECIMAL(3, 2) DEFAULT 0.00,
    
    -- Market Context (for future enhancements)
    market_regime VARCHAR(20),      -- 'trending', 'ranging', 'breakout'
    volatility_regime VARCHAR(20),  -- 'low', 'medium', 'high'
    session_preference VARCHAR(50), -- 'london', 'new_york', 'overlap', 'any'
    
    -- Timestamps
    last_optimized TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional metadata
    optimization_days_used INTEGER,
    total_combinations_tested INTEGER,
    confidence_interval DECIMAL(3, 2) DEFAULT 0.95, -- Statistical confidence
    
    -- Trigger to update last_updated
    CONSTRAINT valid_win_rate CHECK (best_win_rate >= 0 AND best_win_rate <= 100),
    CONSTRAINT valid_profit_factor CHECK (best_profit_factor >= 0),
    CONSTRAINT valid_performance_score CHECK (best_performance_score >= 0)
);

-- Index for efficient epic lookups
CREATE INDEX IF NOT EXISTS idx_smc_best_epic ON smc_best_parameters(epic);
CREATE INDEX IF NOT EXISTS idx_smc_best_score ON smc_best_parameters(best_performance_score);
CREATE INDEX IF NOT EXISTS idx_smc_best_last_optimized ON smc_best_parameters(last_optimized);

-- Trigger to update last_updated timestamp
CREATE OR REPLACE FUNCTION update_smc_best_parameters_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_smc_best_parameters_updated
    BEFORE UPDATE ON smc_best_parameters
    FOR EACH ROW EXECUTE FUNCTION update_smc_best_parameters_updated();

-- =============================================================================
-- HELPFUL VIEWS FOR ANALYSIS
-- =============================================================================

-- View for top performing SMC configurations per epic
CREATE OR REPLACE VIEW smc_top_configurations AS
SELECT DISTINCT ON (epic)
    epic,
    smc_config,
    confidence_level,
    timeframe,
    swing_length,
    structure_confirmation,
    order_block_length,
    fvg_min_size,
    confluence_required,
    min_risk_reward,
    stop_loss_pips,
    take_profit_pips,
    win_rate,
    profit_factor,
    net_pips,
    performance_score,
    structure_breaks_detected,
    order_block_reactions,
    fvg_reactions,
    confluence_accuracy
FROM smc_optimization_results
ORDER BY epic, performance_score DESC;

-- View for SMC optimization summary statistics
CREATE OR REPLACE VIEW smc_optimization_summary AS
SELECT 
    epic,
    COUNT(*) as total_tests,
    AVG(win_rate) as avg_win_rate,
    MAX(win_rate) as max_win_rate,
    AVG(profit_factor) as avg_profit_factor,
    MAX(profit_factor) as max_profit_factor,
    AVG(net_pips) as avg_net_pips,
    MAX(net_pips) as max_net_pips,
    AVG(performance_score) as avg_performance_score,
    MAX(performance_score) as max_performance_score,
    AVG(confluence_accuracy) as avg_confluence_accuracy,
    MAX(confluence_accuracy) as max_confluence_accuracy
FROM smc_optimization_results
GROUP BY epic
ORDER BY max_performance_score DESC;

-- =============================================================================
-- SAMPLE QUERIES FOR ANALYSIS
-- =============================================================================

-- Find best SMC configurations for each epic:
-- SELECT * FROM smc_top_configurations ORDER BY performance_score DESC;

-- Get optimization summary for all epics:
-- SELECT * FROM smc_optimization_summary;

-- Find best configurations by specific criteria:
-- SELECT * FROM smc_optimization_results 
-- WHERE win_rate > 70 AND profit_factor > 2.0 AND confluence_accuracy > 80
-- ORDER BY performance_score DESC LIMIT 20;

-- Analyze parameter impact on performance:
-- SELECT smc_config, AVG(performance_score), COUNT(*)
-- FROM smc_optimization_results
-- GROUP BY smc_config
-- ORDER BY AVG(performance_score) DESC;

-- Get recent optimization runs:
-- SELECT run_id, epic, optimization_mode, status, total_combinations, 
--        completed_combinations, best_score, created_at
-- FROM smc_optimization_runs
-- ORDER BY created_at DESC LIMIT 10;

COMMENT ON TABLE smc_optimization_runs IS 'Tracks SMC strategy optimization sessions and metadata';
COMMENT ON TABLE smc_optimization_results IS 'Stores detailed results for each SMC parameter combination tested';
COMMENT ON TABLE smc_best_parameters IS 'Stores optimal SMC parameters for each epic with performance metrics';
COMMENT ON VIEW smc_top_configurations IS 'Shows best performing SMC configuration for each epic';
COMMENT ON VIEW smc_optimization_summary IS 'Provides summary statistics for SMC optimization results by epic';