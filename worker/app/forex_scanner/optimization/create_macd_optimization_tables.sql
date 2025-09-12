-- MACD Strategy Optimization Database Schema
-- Creates tables for storing MACD optimization runs, results, and best parameters

-- Table 1: MACD Optimization Runs
-- Tracks optimization sessions and metadata
CREATE TABLE IF NOT EXISTS macd_optimization_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(100) NOT NULL,
    description TEXT,
    start_time TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITHOUT TIME ZONE,
    total_combinations INTEGER,
    status VARCHAR(20) DEFAULT 'running'
);

-- Table 2: MACD Optimization Results  
-- Stores individual parameter combination test results
CREATE TABLE IF NOT EXISTS macd_optimization_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES macd_optimization_runs(id),
    epic VARCHAR(50) NOT NULL,
    
    -- Core MACD Parameters
    fast_ema INTEGER NOT NULL,
    slow_ema INTEGER NOT NULL,
    signal_ema INTEGER NOT NULL,
    confidence_threshold NUMERIC(4,3) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    
    -- MACD Validation Filters
    macd_histogram_threshold NUMERIC(10,8) NOT NULL,
    macd_zero_line_filter BOOLEAN DEFAULT FALSE,
    macd_rsi_filter_enabled BOOLEAN DEFAULT FALSE,
    macd_rsi_period INTEGER DEFAULT 14,
    macd_momentum_confirmation BOOLEAN DEFAULT FALSE,
    
    -- Multi-Timeframe Validation
    mtf_enabled BOOLEAN DEFAULT FALSE,
    mtf_timeframes TEXT, -- JSON array of timeframes ['15m', '1h']
    mtf_min_alignment NUMERIC(4,3) DEFAULT 0.6,
    
    -- Strategy Enhancement Options
    smart_money_enabled BOOLEAN DEFAULT FALSE,
    ema_200_trend_filter BOOLEAN DEFAULT TRUE,
    contradiction_filter_enabled BOOLEAN DEFAULT TRUE,
    
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
    
    -- MACD-specific Performance Metrics
    crossover_signals INTEGER DEFAULT 0,
    momentum_confirmed_signals INTEGER DEFAULT 0,
    histogram_strength_avg NUMERIC(10,8),
    false_signal_rate NUMERIC(5,4),
    signal_delay_avg_bars NUMERIC(6,2),
    
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table 3: MACD Best Parameters
-- Stores the optimal parameter set for each epic
CREATE TABLE IF NOT EXISTS macd_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,
    
    -- Best Core MACD Parameters
    best_fast_ema INTEGER NOT NULL,
    best_slow_ema INTEGER NOT NULL,
    best_signal_ema INTEGER NOT NULL,
    best_confidence_threshold NUMERIC(4,3) NOT NULL,
    best_timeframe VARCHAR(5) NOT NULL,
    
    -- Best MACD Filter Configuration
    best_histogram_threshold NUMERIC(10,8) NOT NULL,
    best_zero_line_filter BOOLEAN DEFAULT FALSE,
    best_rsi_filter_enabled BOOLEAN DEFAULT FALSE,
    best_rsi_period INTEGER DEFAULT 14,
    best_momentum_confirmation BOOLEAN DEFAULT FALSE,
    
    -- Best Multi-Timeframe Settings
    best_mtf_enabled BOOLEAN DEFAULT FALSE,
    best_mtf_timeframes TEXT, -- JSON array
    best_mtf_min_alignment NUMERIC(4,3) DEFAULT 0.6,
    
    -- Best Strategy Enhancement Options
    best_smart_money_enabled BOOLEAN DEFAULT FALSE,
    best_ema_200_trend_filter BOOLEAN DEFAULT TRUE,
    best_contradiction_filter_enabled BOOLEAN DEFAULT TRUE,
    
    -- Optimal Risk Management
    optimal_stop_loss_pips NUMERIC(6,1) NOT NULL,
    optimal_take_profit_pips NUMERIC(6,1) NOT NULL,
    
    -- Best Performance Achieved
    best_win_rate NUMERIC(5,4),
    best_profit_factor NUMERIC(8,4),
    best_net_pips NUMERIC(10,2),
    best_composite_score NUMERIC(10,6),
    
    -- MACD-specific Best Performance
    best_crossover_accuracy NUMERIC(5,4),
    best_momentum_confirmation_rate NUMERIC(5,4),
    best_signal_quality_score NUMERIC(8,4),
    
    -- Market Context (for future enhancement)
    market_regime VARCHAR(20),
    session_preference VARCHAR(50),
    volatility_range VARCHAR(20),
    
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_macd_results_epic ON macd_optimization_results(epic);
CREATE INDEX IF NOT EXISTS idx_macd_results_composite_score ON macd_optimization_results(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_macd_results_run_id ON macd_optimization_results(run_id);
CREATE INDEX IF NOT EXISTS idx_macd_results_created_at ON macd_optimization_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_macd_results_win_rate ON macd_optimization_results(win_rate DESC);
CREATE INDEX IF NOT EXISTS idx_macd_results_profit_factor ON macd_optimization_results(profit_factor DESC);

-- Add indexes for MACD-specific queries
CREATE INDEX IF NOT EXISTS idx_macd_results_fast_slow ON macd_optimization_results(fast_ema, slow_ema);
CREATE INDEX IF NOT EXISTS idx_macd_results_timeframe ON macd_optimization_results(timeframe);
CREATE INDEX IF NOT EXISTS idx_macd_results_signals ON macd_optimization_results(total_signals DESC);

-- Add comments for documentation
COMMENT ON TABLE macd_optimization_runs IS 'Tracks MACD strategy optimization sessions';
COMMENT ON TABLE macd_optimization_results IS 'Stores individual parameter combination test results for MACD strategy';
COMMENT ON TABLE macd_best_parameters IS 'Contains optimal parameter sets for each epic using MACD strategy';

COMMENT ON COLUMN macd_optimization_results.fast_ema IS 'Fast EMA period for MACD calculation';
COMMENT ON COLUMN macd_optimization_results.slow_ema IS 'Slow EMA period for MACD calculation';
COMMENT ON COLUMN macd_optimization_results.signal_ema IS 'Signal line EMA period for MACD';
COMMENT ON COLUMN macd_optimization_results.macd_histogram_threshold IS 'Minimum histogram value required for signal validation';
COMMENT ON COLUMN macd_optimization_results.composite_score IS 'Combined performance score (win_rate * profit_factor * net_pips/100)';
COMMENT ON COLUMN macd_optimization_results.mtf_enabled IS 'Multi-timeframe validation enabled flag';
COMMENT ON COLUMN macd_optimization_results.crossover_signals IS 'Number of MACD line crossover signals detected';
COMMENT ON COLUMN macd_optimization_results.momentum_confirmed_signals IS 'Signals that passed momentum confirmation filter';
COMMENT ON COLUMN macd_optimization_results.histogram_strength_avg IS 'Average histogram strength at signal generation';
COMMENT ON COLUMN macd_optimization_results.false_signal_rate IS 'Percentage of signals that resulted in losses';
COMMENT ON COLUMN macd_optimization_results.signal_delay_avg_bars IS 'Average delay in bars from crossover to signal';

-- Display table creation confirmation
SELECT 'MACD optimization tables created successfully!' as status;