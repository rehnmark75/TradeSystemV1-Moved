-- Migration: Create backtest_job_queue table
-- Purpose: Database-backed job queue for triggering backtests from Streamlit
-- Date: 2026-01-09

-- Create job queue table
CREATE TABLE IF NOT EXISTS backtest_job_queue (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(64) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed, cancelled

    -- Backtest configuration
    epic VARCHAR(64) NOT NULL,
    days INTEGER NOT NULL DEFAULT 14,
    strategy VARCHAR(64) DEFAULT 'SMC_SIMPLE',
    timeframe VARCHAR(10) DEFAULT '15m',

    -- Execution options
    parallel BOOLEAN DEFAULT FALSE,
    workers INTEGER DEFAULT 4,
    chunk_days INTEGER DEFAULT 7,
    generate_chart BOOLEAN DEFAULT TRUE,
    pipeline_mode BOOLEAN DEFAULT FALSE,

    -- Parameter overrides (JSON)
    parameter_overrides JSONB DEFAULT '{}',

    -- Snapshot to load
    snapshot_name VARCHAR(128),

    -- Date range (alternative to days)
    start_date DATE,
    end_date DATE,

    -- Job tracking
    submitted_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Result reference
    execution_id INTEGER REFERENCES backtest_executions(id),
    error_message TEXT,

    -- Priority (lower = higher priority)
    priority INTEGER DEFAULT 2,

    -- Submitted by (for audit)
    submitted_by VARCHAR(64) DEFAULT 'streamlit'
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_backtest_job_queue_status ON backtest_job_queue(status);
CREATE INDEX IF NOT EXISTS idx_backtest_job_queue_submitted ON backtest_job_queue(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_job_queue_pending ON backtest_job_queue(status, priority, submitted_at)
    WHERE status = 'pending';

-- Comment
COMMENT ON TABLE backtest_job_queue IS 'Database-backed job queue for backtest submissions from Streamlit';
