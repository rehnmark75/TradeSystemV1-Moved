-- Migration: Create backtest config snapshots table
-- Purpose: Store named parameter configurations for backtesting without affecting live trading
-- Created: 2026-01-06

-- ============================================
-- BACKTEST CONFIG SNAPSHOTS TABLE
-- ============================================
-- Stores named parameter configurations that can be:
-- 1. Created from current config with overrides
-- 2. Used across multiple backtest runs
-- 3. Compared side-by-side
-- 4. Promoted to live trading after validation

CREATE TABLE IF NOT EXISTS smc_backtest_snapshots (
    id SERIAL PRIMARY KEY,

    -- Snapshot identification
    snapshot_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,

    -- Base configuration reference
    base_config_id INTEGER,  -- Reference to smc_simple_global_config if needed
    base_config_version VARCHAR(20),  -- e.g., "v2.10.0"

    -- Parameter overrides (the actual config changes)
    parameter_overrides JSONB NOT NULL DEFAULT '{}',

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'cli',

    -- Test results tracking (populated after backtest runs)
    last_tested_at TIMESTAMP,
    last_test_execution_id INTEGER,
    test_results JSONB,  -- {win_rate, profit_factor, sharpe, trades_count, etc.}
    test_count INTEGER DEFAULT 0,

    -- Promotion tracking
    is_promoted BOOLEAN DEFAULT FALSE,
    promoted_to_live_at TIMESTAMP,
    promoted_by VARCHAR(100),
    promotion_notes TEXT,

    -- Safety flags
    is_backtest_only BOOLEAN DEFAULT TRUE,  -- Prevents accidental live use
    is_active BOOLEAN DEFAULT TRUE,  -- Soft delete

    -- Tags for organization
    tags TEXT[] DEFAULT '{}'
);

-- ============================================
-- INDEXES
-- ============================================
CREATE INDEX IF NOT EXISTS idx_snapshots_name ON smc_backtest_snapshots(snapshot_name);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON smc_backtest_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_is_active ON smc_backtest_snapshots(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_snapshots_is_promoted ON smc_backtest_snapshots(is_promoted) WHERE is_promoted = TRUE;
CREATE INDEX IF NOT EXISTS idx_snapshots_tags ON smc_backtest_snapshots USING GIN(tags);

-- ============================================
-- SNAPSHOT TEST HISTORY TABLE
-- ============================================
-- Track all backtest runs against each snapshot for comparison
CREATE TABLE IF NOT EXISTS smc_snapshot_test_history (
    id SERIAL PRIMARY KEY,
    snapshot_id INTEGER REFERENCES smc_backtest_snapshots(id) ON DELETE CASCADE,
    execution_id INTEGER NOT NULL,  -- Reference to backtest_executions

    -- Test configuration
    epic_tested VARCHAR(50),
    days_tested INTEGER,
    start_date TIMESTAMP,
    end_date TIMESTAMP,

    -- Results
    total_signals INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(6,3),
    total_pips DECIMAL(10,2),
    avg_profit_pips DECIMAL(6,2),
    avg_loss_pips DECIMAL(6,2),
    max_drawdown_pips DECIMAL(8,2),
    sharpe_ratio DECIMAL(6,3),

    -- Exit breakdown
    profit_target_exits INTEGER DEFAULT 0,
    stop_loss_exits INTEGER DEFAULT 0,
    trailing_stop_exits INTEGER DEFAULT 0,

    -- Metadata
    tested_at TIMESTAMP DEFAULT NOW(),
    test_duration_seconds DECIMAL(8,2),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshot_test_history_snapshot ON smc_snapshot_test_history(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_snapshot_test_history_tested_at ON smc_snapshot_test_history(tested_at DESC);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_snapshot_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update timestamp
DROP TRIGGER IF EXISTS trigger_update_snapshot_timestamp ON smc_backtest_snapshots;
CREATE TRIGGER trigger_update_snapshot_timestamp
    BEFORE UPDATE ON smc_backtest_snapshots
    FOR EACH ROW
    EXECUTE FUNCTION update_snapshot_timestamp();

-- Function to get snapshot with merged parameters
CREATE OR REPLACE FUNCTION get_snapshot_full_config(p_snapshot_name VARCHAR)
RETURNS JSONB AS $$
DECLARE
    v_overrides JSONB;
    v_base_config JSONB;
BEGIN
    -- Get snapshot overrides
    SELECT parameter_overrides INTO v_overrides
    FROM smc_backtest_snapshots
    WHERE snapshot_name = p_snapshot_name AND is_active = TRUE;

    IF v_overrides IS NULL THEN
        RETURN NULL;
    END IF;

    -- Get current base config from smc_simple_global_config
    SELECT row_to_json(c)::jsonb INTO v_base_config
    FROM smc_simple_global_config c
    WHERE is_active = TRUE
    LIMIT 1;

    -- Merge base config with overrides (overrides take precedence)
    RETURN v_base_config || v_overrides;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- INITIAL DATA / EXAMPLE SNAPSHOTS
-- ============================================
-- These are commented out - use snapshot_cli.py to create snapshots

-- Example: Tight stop loss configuration
-- INSERT INTO smc_backtest_snapshots (snapshot_name, description, parameter_overrides, tags)
-- VALUES (
--     'tight_sl_test',
--     'Testing tighter stop loss for reduced risk',
--     '{"fixed_stop_loss_pips": 8, "sl_buffer_pips": 1}',
--     ARRAY['sl_optimization', 'risk_reduction']
-- );

-- Example: High confidence filter
-- INSERT INTO smc_backtest_snapshots (snapshot_name, description, parameter_overrides, tags)
-- VALUES (
--     'high_confidence',
--     'Only take high confidence signals',
--     '{"min_confidence": 0.65, "max_confidence": 0.95}',
--     ARRAY['confidence', 'quality_filter']
-- );

-- ============================================
-- VERIFICATION
-- ============================================
DO $$
BEGIN
    RAISE NOTICE '✅ Backtest config snapshots tables created successfully';
    RAISE NOTICE '   - smc_backtest_snapshots: Named parameter configurations';
    RAISE NOTICE '   - smc_snapshot_test_history: Test results tracking';
    RAISE NOTICE '   - Use snapshot_cli.py to manage snapshots';
END $$;
