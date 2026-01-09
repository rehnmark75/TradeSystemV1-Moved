-- Migration: Create parallel backtest tracking table
-- Purpose: Track chunked parallel backtest runs and link to existing backtest_executions
-- Date: 2026-01-09

-- Create the parallel runs tracking table
CREATE TABLE IF NOT EXISTS backtest_parallel_runs (
    id SERIAL PRIMARY KEY,

    -- Master run info
    epic VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    full_start_date TIMESTAMP NOT NULL,
    full_end_date TIMESTAMP NOT NULL,

    -- Chunk configuration
    chunk_days INTEGER DEFAULT 7,
    warmup_days INTEGER DEFAULT 2,
    worker_count INTEGER DEFAULT 4,

    -- Progress tracking
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'aggregating', 'completed', 'failed', 'cancelled')),
    total_chunks INTEGER NOT NULL,
    completed_chunks INTEGER DEFAULT 0,

    -- Linked chunk executions (array of backtest_executions.id)
    chunk_execution_ids INTEGER[] DEFAULT '{}',

    -- Aggregated results (computed after all chunks complete)
    aggregated_results JSONB,

    -- Error tracking
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Original config for reproduction
    config_snapshot JSONB
);

-- Index for status queries (find running/pending jobs)
CREATE INDEX IF NOT EXISTS idx_parallel_runs_status ON backtest_parallel_runs(status);

-- Index for epic queries (find runs by pair)
CREATE INDEX IF NOT EXISTS idx_parallel_runs_epic ON backtest_parallel_runs(epic);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_parallel_runs_dates ON backtest_parallel_runs(full_start_date, full_end_date);

-- Index for created_at (recent runs)
CREATE INDEX IF NOT EXISTS idx_parallel_runs_created ON backtest_parallel_runs(created_at DESC);

-- Comment on table
COMMENT ON TABLE backtest_parallel_runs IS 'Tracks parallel chunked backtest runs, linking multiple backtest_executions into a single logical run';

-- Comment on columns
COMMENT ON COLUMN backtest_parallel_runs.chunk_execution_ids IS 'Array of backtest_executions.id for each chunk in this parallel run';
COMMENT ON COLUMN backtest_parallel_runs.aggregated_results IS 'Combined metrics from all chunks: total_signals, win_rate, total_pips, max_drawdown, etc.';
COMMENT ON COLUMN backtest_parallel_runs.config_snapshot IS 'Full configuration at run time for reproducibility';

-- Helper function to update completed_chunks count
CREATE OR REPLACE FUNCTION update_parallel_run_progress(
    p_run_id INTEGER,
    p_chunk_execution_id INTEGER
) RETURNS VOID AS $$
BEGIN
    UPDATE backtest_parallel_runs
    SET
        chunk_execution_ids = array_append(chunk_execution_ids, p_chunk_execution_id),
        completed_chunks = completed_chunks + 1,
        status = CASE
            WHEN completed_chunks + 1 >= total_chunks THEN 'aggregating'
            ELSE 'running'
        END
    WHERE id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- Helper function to finalize a parallel run with aggregated results
CREATE OR REPLACE FUNCTION finalize_parallel_run(
    p_run_id INTEGER,
    p_aggregated_results JSONB
) RETURNS VOID AS $$
BEGIN
    UPDATE backtest_parallel_runs
    SET
        status = 'completed',
        aggregated_results = p_aggregated_results,
        completed_at = NOW()
    WHERE id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- Helper function to mark a parallel run as failed
CREATE OR REPLACE FUNCTION fail_parallel_run(
    p_run_id INTEGER,
    p_error_message TEXT
) RETURNS VOID AS $$
BEGIN
    UPDATE backtest_parallel_runs
    SET
        status = 'failed',
        error_message = p_error_message,
        completed_at = NOW()
    WHERE id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- Verification query
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'backtest_parallel_runs') THEN
        RAISE NOTICE 'SUCCESS: backtest_parallel_runs table created successfully';
    ELSE
        RAISE EXCEPTION 'FAILED: backtest_parallel_runs table was not created';
    END IF;
END $$;
