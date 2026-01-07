-- Migration: Create multi-epic optimization tables
-- Purpose: Store optimization run metadata, results, and best parameters for multi-epic parameter sweeps
-- Created: 2026-01-07

-- ============================================
-- OPTIMIZATION RUNS TABLE (Master record)
-- ============================================
-- Tracks each optimization run with progress for resume capability

CREATE TABLE IF NOT EXISTS optimization_runs (
    id SERIAL PRIMARY KEY,

    -- Run identification
    run_name VARCHAR(100) UNIQUE NOT NULL,  -- e.g., 'multiepic_20260107_1430_extended'
    run_mode VARCHAR(20) NOT NULL,           -- 'fast', 'medium', 'extended'

    -- Configuration
    epics_to_test TEXT[] NOT NULL,           -- Array of epics to optimize
    days_tested INTEGER NOT NULL DEFAULT 30,
    start_date TIMESTAMP NOT NULL,           -- Backtest start date
    end_date TIMESTAMP NOT NULL,             -- Backtest end date

    -- Parameter grid (JSON for flexibility)
    parameter_grid JSONB NOT NULL,           -- Full parameter grid used
    total_combinations INTEGER NOT NULL,     -- Total tests planned (combinations × epics)

    -- Progress tracking (for resume capability)
    status VARCHAR(20) DEFAULT 'pending',    -- pending, running, paused, completed, failed
    completed_combinations INTEGER DEFAULT 0,
    current_epic VARCHAR(50),                -- Epic currently being tested
    current_epic_idx INTEGER DEFAULT 0,      -- Index in epics array
    current_param_idx INTEGER DEFAULT 0,     -- Index in combinations array

    -- Timing
    started_at TIMESTAMP,
    paused_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Results summary (populated on completion)
    best_overall_epic VARCHAR(50),
    best_overall_params JSONB,
    best_overall_score DECIMAL(10,6),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'cli',
    notes TEXT,

    CONSTRAINT valid_optimization_run_status CHECK (
        status IN ('pending', 'running', 'paused', 'completed', 'failed')
    )
);

-- Indexes for optimization_runs
CREATE INDEX IF NOT EXISTS idx_optimization_runs_status ON optimization_runs(status);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_created ON optimization_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_mode ON optimization_runs(run_mode);


-- ============================================
-- OPTIMIZATION RESULTS TABLE (Individual tests)
-- ============================================
-- Stores result for each epic/parameter combination tested

CREATE TABLE IF NOT EXISTS optimization_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id) ON DELETE CASCADE,

    -- Test identification
    epic VARCHAR(50) NOT NULL,
    execution_id INTEGER,                    -- Reference to backtest_executions.id

    -- Parameters tested (full parameter set as JSON)
    params_tested JSONB NOT NULL,
    rr_ratio DECIMAL(4,2),                   -- Calculated Risk:Reward ratio

    -- Performance metrics
    total_signals INTEGER DEFAULT 0,
    winners INTEGER DEFAULT 0,
    losers INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),                   -- 0.0000 to 1.0000
    profit_factor DECIMAL(8,3),
    total_pips DECIMAL(10,2),
    avg_profit_pips DECIMAL(8,2),
    avg_loss_pips DECIMAL(8,2),
    max_drawdown_pips DECIMAL(10,2),

    -- Composite scoring (for ranking)
    composite_score DECIMAL(10,6),

    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending',    -- pending, running, completed, failed
    error_message TEXT,
    duration_seconds DECIMAL(8,2),

    -- Metadata
    tested_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_optimization_result_status CHECK (
        status IN ('pending', 'running', 'completed', 'failed')
    )
);

-- Indexes for optimization_results
CREATE INDEX IF NOT EXISTS idx_optimization_results_run ON optimization_results(run_id);
CREATE INDEX IF NOT EXISTS idx_optimization_results_epic ON optimization_results(epic);
CREATE INDEX IF NOT EXISTS idx_optimization_results_score ON optimization_results(composite_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_optimization_results_run_epic ON optimization_results(run_id, epic);


-- ============================================
-- OPTIMIZATION BEST PARAMS TABLE (Summary)
-- ============================================
-- Quick lookup table for best parameters per epic per run

CREATE TABLE IF NOT EXISTS optimization_best_params (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id) ON DELETE CASCADE,
    epic VARCHAR(50) NOT NULL,

    -- Best parameters found
    best_params JSONB NOT NULL,

    -- Performance metrics for best config
    total_signals INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,3),
    total_pips DECIMAL(10,2),
    composite_score DECIMAL(10,6),

    -- Snapshot reference (if auto-created)
    snapshot_name VARCHAR(100),
    snapshot_id INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),

    -- Each run should have only one best per epic
    UNIQUE (run_id, epic)
);

-- Indexes for optimization_best_params
CREATE INDEX IF NOT EXISTS idx_optimization_best_epic ON optimization_best_params(epic);
CREATE INDEX IF NOT EXISTS idx_optimization_best_run ON optimization_best_params(run_id);
CREATE INDEX IF NOT EXISTS idx_optimization_best_score ON optimization_best_params(composite_score DESC NULLS LAST);


-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Auto-update timestamp on optimization_runs
CREATE OR REPLACE FUNCTION update_optimization_runs_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_optimization_runs_timestamp ON optimization_runs;
CREATE TRIGGER trigger_update_optimization_runs_timestamp
    BEFORE UPDATE ON optimization_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_optimization_runs_timestamp();


-- Function to get best params for an epic across all runs
CREATE OR REPLACE FUNCTION get_best_params_for_epic(p_epic VARCHAR)
RETURNS TABLE (
    run_id INTEGER,
    run_name VARCHAR,
    best_params JSONB,
    win_rate DECIMAL,
    profit_factor DECIMAL,
    total_pips DECIMAL,
    composite_score DECIMAL,
    tested_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        bp.run_id,
        r.run_name,
        bp.best_params,
        bp.win_rate,
        bp.profit_factor,
        bp.total_pips,
        bp.composite_score,
        bp.created_at as tested_at
    FROM optimization_best_params bp
    JOIN optimization_runs r ON bp.run_id = r.id
    WHERE bp.epic = p_epic
    ORDER BY bp.composite_score DESC
    LIMIT 10;
END;
$$ LANGUAGE plpgsql;


-- Function to get run progress summary
CREATE OR REPLACE FUNCTION get_optimization_run_progress(p_run_id INTEGER)
RETURNS TABLE (
    epic VARCHAR,
    tests_completed BIGINT,
    tests_total BIGINT,
    best_score DECIMAL,
    avg_win_rate DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.epic,
        COUNT(CASE WHEN r.status = 'completed' THEN 1 END) as tests_completed,
        COUNT(*) as tests_total,
        MAX(r.composite_score) as best_score,
        AVG(r.win_rate) as avg_win_rate
    FROM optimization_results r
    WHERE r.run_id = p_run_id
    GROUP BY r.epic
    ORDER BY r.epic;
END;
$$ LANGUAGE plpgsql;


-- ============================================
-- VERIFICATION
-- ============================================
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Multi-Epic Optimization Tables Created';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables:';
    RAISE NOTICE '  - optimization_runs: Master run tracking with resume capability';
    RAISE NOTICE '  - optimization_results: Individual test results per epic/params';
    RAISE NOTICE '  - optimization_best_params: Best params summary per epic';
    RAISE NOTICE '';
    RAISE NOTICE 'Functions:';
    RAISE NOTICE '  - get_best_params_for_epic(epic): Get best params across runs';
    RAISE NOTICE '  - get_optimization_run_progress(run_id): Get run progress';
    RAISE NOTICE '========================================';
END $$;
