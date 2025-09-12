-- Zero-Lag Strategy Optimization Database Schema
-- Creates tables for storing zero-lag optimization runs, results, and best parameters

-- Table 1: Zero-Lag Optimization Runs
-- Tracks optimization sessions and metadata
CREATE TABLE IF NOT EXISTS zerolag_optimization_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(100) NOT NULL,
    description TEXT,
    start_time TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITHOUT TIME ZONE,
    total_combinations INTEGER,
    status VARCHAR(20) DEFAULT 'running'
);

-- Table 2: Zero-Lag Optimization Results  
-- Stores individual parameter combination test results
CREATE TABLE IF NOT EXISTS zerolag_optimization_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES zerolag_optimization_runs(id),
    epic VARCHAR(50) NOT NULL,
    
    -- Core Zero-Lag Parameters
    zl_length INTEGER NOT NULL,
    band_multiplier NUMERIC(4,2) NOT NULL,
    confidence_threshold NUMERIC(4,3) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    
    -- Squeeze Momentum Parameters
    bb_length INTEGER NOT NULL,
    bb_mult NUMERIC(4,2) NOT NULL,
    kc_length INTEGER NOT NULL,
    kc_mult NUMERIC(4,2) NOT NULL,
    
    -- Strategy Options
    smart_money_enabled BOOLEAN DEFAULT FALSE,
    mtf_validation_enabled BOOLEAN DEFAULT FALSE,
    
    -- Risk Management
    stop_loss_pips NUMERIC(6,1) NOT NULL,
    take_profit_pips NUMERIC(6,1) NOT NULL,
    risk_reward_ratio NUMERIC(6,3) NOT NULL,
    
    -- Performance Metrics
    total_signals INTEGER DEFAULT 0,
    win_rate NUMERIC(5,4),
    profit_factor NUMERIC(8,4),
    net_pips NUMERIC(10,2),
    composite_score NUMERIC(10,6),
    
    -- Additional Performance Metrics
    avg_profit_pips NUMERIC(8,2),
    avg_loss_pips NUMERIC(8,2),
    total_profit_pips NUMERIC(10,2),
    total_loss_pips NUMERIC(10,2),
    expectancy_per_trade NUMERIC(8,4),
    profit_target_exits INTEGER DEFAULT 0,
    stop_loss_exits INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table 3: Zero-Lag Best Parameters
-- Stores the optimal parameter set for each epic
CREATE TABLE IF NOT EXISTS zerolag_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,
    
    -- Best Core Parameters
    best_zl_length INTEGER NOT NULL,
    best_band_multiplier NUMERIC(4,2) NOT NULL,
    best_confidence_threshold NUMERIC(4,3) NOT NULL,
    best_timeframe VARCHAR(5) NOT NULL,
    
    -- Best Squeeze Parameters
    best_bb_length INTEGER NOT NULL,
    best_bb_mult NUMERIC(4,2) NOT NULL,
    best_kc_length INTEGER NOT NULL,
    best_kc_mult NUMERIC(4,2) NOT NULL,
    
    -- Best Strategy Options
    best_smart_money_enabled BOOLEAN DEFAULT FALSE,
    best_mtf_validation_enabled BOOLEAN DEFAULT FALSE,
    
    -- Optimal Risk Management
    optimal_stop_loss_pips NUMERIC(6,1) NOT NULL,
    optimal_take_profit_pips NUMERIC(6,1) NOT NULL,
    
    -- Best Performance Achieved
    best_win_rate NUMERIC(5,4),
    best_profit_factor NUMERIC(8,4),
    best_net_pips NUMERIC(10,2),
    best_composite_score NUMERIC(10,6),
    
    -- Market Context (for future enhancement)
    market_regime VARCHAR(20),
    session_preference VARCHAR(50),
    volatility_range VARCHAR(20),
    
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_zerolag_results_epic ON zerolag_optimization_results(epic);
CREATE INDEX IF NOT EXISTS idx_zerolag_results_composite_score ON zerolag_optimization_results(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_zerolag_results_run_id ON zerolag_optimization_results(run_id);
CREATE INDEX IF NOT EXISTS idx_zerolag_results_created_at ON zerolag_optimization_results(created_at DESC);

-- Add comments for documentation
COMMENT ON TABLE zerolag_optimization_runs IS 'Tracks zero-lag strategy optimization sessions';
COMMENT ON TABLE zerolag_optimization_results IS 'Stores individual parameter combination test results for zero-lag strategy';
COMMENT ON TABLE zerolag_best_parameters IS 'Contains optimal parameter sets for each epic using zero-lag strategy';

COMMENT ON COLUMN zerolag_optimization_results.zl_length IS 'Zero-lag EMA length parameter';
COMMENT ON COLUMN zerolag_optimization_results.band_multiplier IS 'Volatility band multiplier';
COMMENT ON COLUMN zerolag_optimization_results.bb_length IS 'Bollinger Bands length for squeeze momentum';
COMMENT ON COLUMN zerolag_optimization_results.kc_length IS 'Keltner Channel length for squeeze momentum';
COMMENT ON COLUMN zerolag_optimization_results.composite_score IS 'Combined performance score (win_rate * profit_factor * net_pips/100)';
COMMENT ON COLUMN zerolag_optimization_results.mtf_validation_enabled IS 'Multi-timeframe validation enabled flag';

-- Display table creation confirmation
SELECT 'Zero-Lag optimization tables created successfully!' as status;