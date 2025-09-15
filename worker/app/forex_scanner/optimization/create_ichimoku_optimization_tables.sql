-- Ichimoku Strategy Optimization Database Schema
-- Creates tables for storing Ichimoku optimization runs, results, and best parameters

-- Table 1: Ichimoku Optimization Runs
-- Tracks optimization sessions and metadata
CREATE TABLE IF NOT EXISTS ichimoku_optimization_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(100) NOT NULL,
    description TEXT,
    start_time TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITHOUT TIME ZONE,
    total_combinations INTEGER,
    status VARCHAR(20) DEFAULT 'running'
);

-- Table 2: Ichimoku Optimization Results
-- Stores individual parameter combination test results
CREATE TABLE IF NOT EXISTS ichimoku_optimization_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES ichimoku_optimization_runs(id),
    epic VARCHAR(50) NOT NULL,

    -- Core Ichimoku Parameters (Traditional: 9-26-52-26)
    tenkan_period INTEGER NOT NULL DEFAULT 9,      -- Conversion line period
    kijun_period INTEGER NOT NULL DEFAULT 26,      -- Base line period
    senkou_b_period INTEGER NOT NULL DEFAULT 52,   -- Leading span B period
    chikou_shift INTEGER NOT NULL DEFAULT 26,      -- Lagging span displacement
    cloud_shift INTEGER NOT NULL DEFAULT 26,       -- Cloud forward displacement
    confidence_threshold NUMERIC(4,3) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,

    -- Ichimoku Validation Filters
    cloud_thickness_threshold NUMERIC(10,8) NOT NULL DEFAULT 0.0001,
    tk_cross_strength_threshold NUMERIC(6,4) NOT NULL DEFAULT 0.5,
    chikou_clear_threshold NUMERIC(10,8) NOT NULL DEFAULT 0.0002,
    cloud_filter_enabled BOOLEAN DEFAULT TRUE,
    chikou_filter_enabled BOOLEAN DEFAULT TRUE,
    tk_filter_enabled BOOLEAN DEFAULT TRUE,

    -- Multi-Timeframe Validation
    mtf_enabled BOOLEAN DEFAULT FALSE,
    mtf_timeframes TEXT, -- JSON array of timeframes ['15m', '1h', '4h']
    mtf_min_alignment NUMERIC(4,3) DEFAULT 0.6,
    mtf_cloud_weight NUMERIC(4,3) DEFAULT 0.4,
    mtf_tk_weight NUMERIC(4,3) DEFAULT 0.3,
    mtf_chikou_weight NUMERIC(4,3) DEFAULT 0.3,

    -- Strategy Enhancement Options
    momentum_confluence_enabled BOOLEAN DEFAULT FALSE,
    smart_money_enabled BOOLEAN DEFAULT FALSE,
    ema_200_trend_filter BOOLEAN DEFAULT FALSE,
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

    -- Ichimoku-specific Performance Metrics
    tk_cross_signals INTEGER DEFAULT 0,
    cloud_breakout_signals INTEGER DEFAULT 0,
    chikou_confirmed_signals INTEGER DEFAULT 0,
    perfect_alignment_signals INTEGER DEFAULT 0,
    cloud_thickness_avg NUMERIC(10,8),
    false_breakout_rate NUMERIC(5,4),
    signal_delay_avg_bars NUMERIC(6,2),

    -- Cloud Analysis Metrics
    bull_cloud_accuracy NUMERIC(5,4),
    bear_cloud_accuracy NUMERIC(5,4),
    tk_cross_accuracy NUMERIC(5,4),
    chikou_confirmation_rate NUMERIC(5,4),
    mtf_alignment_avg NUMERIC(5,4),

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Table 3: Ichimoku Best Parameters
-- Stores the optimal parameter set for each epic
CREATE TABLE IF NOT EXISTS ichimoku_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,

    -- Best Core Ichimoku Parameters
    best_tenkan_period INTEGER NOT NULL DEFAULT 9,
    best_kijun_period INTEGER NOT NULL DEFAULT 26,
    best_senkou_b_period INTEGER NOT NULL DEFAULT 52,
    best_chikou_shift INTEGER NOT NULL DEFAULT 26,
    best_cloud_shift INTEGER NOT NULL DEFAULT 26,
    best_confidence_threshold NUMERIC(4,3) NOT NULL,
    best_timeframe VARCHAR(5) NOT NULL,

    -- Best Ichimoku Filter Configuration
    best_cloud_thickness_threshold NUMERIC(10,8) NOT NULL,
    best_tk_cross_strength_threshold NUMERIC(6,4) NOT NULL,
    best_chikou_clear_threshold NUMERIC(10,8) NOT NULL,
    best_cloud_filter_enabled BOOLEAN DEFAULT TRUE,
    best_chikou_filter_enabled BOOLEAN DEFAULT TRUE,
    best_tk_filter_enabled BOOLEAN DEFAULT TRUE,

    -- Best Multi-Timeframe Settings
    best_mtf_enabled BOOLEAN DEFAULT FALSE,
    best_mtf_timeframes TEXT, -- JSON array
    best_mtf_min_alignment NUMERIC(4,3) DEFAULT 0.6,
    best_mtf_cloud_weight NUMERIC(4,3) DEFAULT 0.4,
    best_mtf_tk_weight NUMERIC(4,3) DEFAULT 0.3,
    best_mtf_chikou_weight NUMERIC(4,3) DEFAULT 0.3,

    -- Best Strategy Enhancement Options
    best_momentum_confluence_enabled BOOLEAN DEFAULT FALSE,
    best_smart_money_enabled BOOLEAN DEFAULT FALSE,
    best_ema_200_trend_filter BOOLEAN DEFAULT FALSE,
    best_contradiction_filter_enabled BOOLEAN DEFAULT TRUE,

    -- Optimal Risk Management
    optimal_stop_loss_pips NUMERIC(6,1) NOT NULL,
    optimal_take_profit_pips NUMERIC(6,1) NOT NULL,

    -- Best Performance Achieved
    best_win_rate NUMERIC(5,4),
    best_profit_factor NUMERIC(8,4),
    best_net_pips NUMERIC(10,2),
    best_composite_score NUMERIC(10,6),

    -- Ichimoku-specific Best Performance
    best_tk_cross_accuracy NUMERIC(5,4),
    best_cloud_breakout_accuracy NUMERIC(5,4),
    best_chikou_confirmation_rate NUMERIC(5,4),
    best_perfect_alignment_rate NUMERIC(5,4),
    best_signal_quality_score NUMERIC(8,4),
    best_mtf_alignment_avg NUMERIC(5,4),

    -- Market Context (for future enhancement)
    market_regime VARCHAR(20) DEFAULT 'trending',
    session_preference VARCHAR(50) DEFAULT 'all_sessions',
    volatility_range VARCHAR(20) DEFAULT 'medium',

    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_epic ON ichimoku_optimization_results(epic);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_composite_score ON ichimoku_optimization_results(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_run_id ON ichimoku_optimization_results(run_id);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_created_at ON ichimoku_optimization_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_win_rate ON ichimoku_optimization_results(win_rate DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_profit_factor ON ichimoku_optimization_results(profit_factor DESC);

-- Add indexes for Ichimoku-specific queries
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_tk_periods ON ichimoku_optimization_results(tenkan_period, kijun_period);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_timeframe ON ichimoku_optimization_results(timeframe);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_signals ON ichimoku_optimization_results(total_signals DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_tk_signals ON ichimoku_optimization_results(tk_cross_signals DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_cloud_signals ON ichimoku_optimization_results(cloud_breakout_signals DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_perfect_alignment ON ichimoku_optimization_results(perfect_alignment_signals DESC);

-- Add indexes for performance analysis
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_tk_accuracy ON ichimoku_optimization_results(tk_cross_accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_cloud_accuracy ON ichimoku_optimization_results(bull_cloud_accuracy DESC, bear_cloud_accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_ichimoku_results_chikou_rate ON ichimoku_optimization_results(chikou_confirmation_rate DESC);

-- Add comments for documentation
COMMENT ON TABLE ichimoku_optimization_runs IS 'Tracks Ichimoku strategy optimization sessions';
COMMENT ON TABLE ichimoku_optimization_results IS 'Stores individual parameter combination test results for Ichimoku strategy';
COMMENT ON TABLE ichimoku_best_parameters IS 'Contains optimal parameter sets for each epic using Ichimoku strategy';

COMMENT ON COLUMN ichimoku_optimization_results.tenkan_period IS 'Conversion line (Tenkan-sen) period for high/low midpoint calculation';
COMMENT ON COLUMN ichimoku_optimization_results.kijun_period IS 'Base line (Kijun-sen) period for high/low midpoint calculation';
COMMENT ON COLUMN ichimoku_optimization_results.senkou_b_period IS 'Leading Span B period for cloud calculation';
COMMENT ON COLUMN ichimoku_optimization_results.chikou_shift IS 'Lagging span displacement periods (backward shift)';
COMMENT ON COLUMN ichimoku_optimization_results.cloud_shift IS 'Cloud forward displacement periods (future projection)';
COMMENT ON COLUMN ichimoku_optimization_results.cloud_thickness_threshold IS 'Minimum cloud thickness ratio for reliable signals';
COMMENT ON COLUMN ichimoku_optimization_results.tk_cross_strength_threshold IS 'Minimum TK line separation for strong crossover signals';
COMMENT ON COLUMN ichimoku_optimization_results.chikou_clear_threshold IS 'Minimum clearance for Chikou span validation';
COMMENT ON COLUMN ichimoku_optimization_results.composite_score IS 'Combined performance score (win_rate * profit_factor * net_pips/100)';
COMMENT ON COLUMN ichimoku_optimization_results.mtf_enabled IS 'Multi-timeframe validation enabled flag';
COMMENT ON COLUMN ichimoku_optimization_results.tk_cross_signals IS 'Number of Tenkan-Kijun crossover signals detected';
COMMENT ON COLUMN ichimoku_optimization_results.cloud_breakout_signals IS 'Number of cloud breakout signals detected';
COMMENT ON COLUMN ichimoku_optimization_results.chikou_confirmed_signals IS 'Signals confirmed by Chikou span analysis';
COMMENT ON COLUMN ichimoku_optimization_results.perfect_alignment_signals IS 'Signals with perfect Ichimoku component alignment';
COMMENT ON COLUMN ichimoku_optimization_results.cloud_thickness_avg IS 'Average cloud thickness at signal generation';
COMMENT ON COLUMN ichimoku_optimization_results.false_breakout_rate IS 'Percentage of cloud breakouts that were false signals';
COMMENT ON COLUMN ichimoku_optimization_results.signal_delay_avg_bars IS 'Average delay in bars from crossover to signal';
COMMENT ON COLUMN ichimoku_optimization_results.bull_cloud_accuracy IS 'Accuracy of bullish cloud breakout signals';
COMMENT ON COLUMN ichimoku_optimization_results.bear_cloud_accuracy IS 'Accuracy of bearish cloud breakout signals';
COMMENT ON COLUMN ichimoku_optimization_results.tk_cross_accuracy IS 'Accuracy of Tenkan-Kijun crossover signals';
COMMENT ON COLUMN ichimoku_optimization_results.chikou_confirmation_rate IS 'Rate of signals confirmed by Chikou span';
COMMENT ON COLUMN ichimoku_optimization_results.mtf_alignment_avg IS 'Average multi-timeframe alignment score';

-- Display table creation confirmation
SELECT 'Ichimoku optimization tables created successfully!' as status;