-- =============================================================================
-- BACKTEST INTEGRATION DATABASE SCHEMA
-- Migration: 001_create_backtest_tables.sql
-- Description: Create database tables for backtest integration with scanner pipeline
-- =============================================================================

-- =============================================================================
-- BACKTEST EXECUTION TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_executions (
    id SERIAL PRIMARY KEY,
    execution_name VARCHAR(100) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,

    -- Backtest Configuration
    data_start_date TIMESTAMP NOT NULL,
    data_end_date TIMESTAMP NOT NULL,
    epics_tested TEXT[] NOT NULL,
    timeframes TEXT[] NOT NULL,

    -- Execution Status
    status VARCHAR(20) DEFAULT 'running',
    total_combinations INTEGER,
    completed_combinations INTEGER DEFAULT 0,
    error_message TEXT,

    -- Data Quality Metrics
    total_candles_processed BIGINT DEFAULT 0,
    data_gaps_detected INTEGER DEFAULT 0,
    quality_score DECIMAL(3,2) DEFAULT 1.0,

    -- Performance Metadata
    execution_duration_seconds INTEGER,
    memory_usage_mb INTEGER,

    -- Configuration snapshot
    config_snapshot JSONB,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_backtest_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- =============================================================================
-- BACKTEST SIGNAL RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_signals (
    id BIGSERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Signal Identification
    epic VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_timestamp TIMESTAMP NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'BULL', 'BEAR'
    strategy_name VARCHAR(50) NOT NULL,

    -- Market Data Context
    open_price DECIMAL(12,6) NOT NULL,
    high_price DECIMAL(12,6) NOT NULL,
    low_price DECIMAL(12,6) NOT NULL,
    close_price DECIMAL(12,6) NOT NULL,
    volume BIGINT,

    -- Signal Characteristics
    confidence_score DECIMAL(5,4) NOT NULL,
    signal_strength DECIMAL(5,4),

    -- Technical Indicators (JSON for flexibility)
    indicator_values JSONB,

    -- Trade Execution Parameters
    entry_price DECIMAL(12,6) NOT NULL,
    stop_loss_price DECIMAL(12,6),
    take_profit_price DECIMAL(12,6),
    risk_reward_ratio DECIMAL(6,3),

    -- Trade Outcome (null if trade not completed in backtest period)
    exit_price DECIMAL(12,6),
    exit_timestamp TIMESTAMP,
    exit_reason VARCHAR(30), -- 'take_profit', 'stop_loss', 'timeout', 'reversal'
    pips_gained DECIMAL(8,2),
    trade_result VARCHAR(10), -- 'win', 'loss', 'breakeven'

    -- Performance Metrics
    holding_time_minutes INTEGER,
    max_favorable_excursion_pips DECIMAL(8,2),
    max_adverse_excursion_pips DECIMAL(8,2),

    -- Data Quality
    data_completeness DECIMAL(3,2) DEFAULT 1.0,
    validation_flags TEXT[],

    -- Validation Pipeline Results (same as live trading)
    validation_passed BOOLEAN DEFAULT FALSE,
    validation_reasons TEXT[],
    trade_validator_version VARCHAR(20),

    -- Market Intelligence (same as live signals)
    market_intelligence JSONB,
    smart_money_score DECIMAL(5,4),
    smart_money_validated BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_signal_type CHECK (signal_type IN ('BULL', 'BEAR')),
    CONSTRAINT valid_trade_result CHECK (trade_result IS NULL OR trade_result IN ('win', 'loss', 'breakeven'))
);

-- =============================================================================
-- BACKTEST PERFORMANCE AGGREGATES
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_performance (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,
    epic VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,

    -- Trading Metrics
    total_signals INTEGER NOT NULL DEFAULT 0,
    bull_signals INTEGER NOT NULL DEFAULT 0,
    bear_signals INTEGER NOT NULL DEFAULT 0,
    validated_signals INTEGER NOT NULL DEFAULT 0,

    -- Win/Loss Analysis
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    breakeven_trades INTEGER NOT NULL DEFAULT 0,
    win_rate DECIMAL(5,4),

    -- Profit/Loss Metrics
    total_pips DECIMAL(10,2) DEFAULT 0,
    avg_win_pips DECIMAL(8,2),
    avg_loss_pips DECIMAL(8,2),
    max_win_pips DECIMAL(8,2),
    max_loss_pips DECIMAL(8,2),

    -- Risk Metrics
    profit_factor DECIMAL(8,4),
    expectancy_per_trade DECIMAL(8,3),
    max_drawdown_pips DECIMAL(8,2),
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,

    -- Time-based Analysis
    avg_trade_duration_minutes DECIMAL(8,2),
    signal_frequency_per_day DECIMAL(6,3),

    -- Data Quality
    data_completeness_score DECIMAL(3,2) DEFAULT 1.0,
    missing_candles_count INTEGER DEFAULT 0,

    -- Comparison with Live Performance
    live_correlation_score DECIMAL(3,2),
    statistical_significance DECIMAL(5,4),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_backtest_performance UNIQUE(execution_id, epic, timeframe, strategy_name)
);

-- =============================================================================
-- OPTIMIZED INDEXES FOR BACKTEST QUERIES
-- =============================================================================

-- Time-series queries (most common access pattern)
CREATE INDEX IF NOT EXISTS idx_backtest_signals_epic_time ON backtest_signals(epic, signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_signals_timeframe_time ON backtest_signals(timeframe, signal_timestamp DESC);

-- Strategy performance analysis
CREATE INDEX IF NOT EXISTS idx_backtest_signals_strategy_epic ON backtest_signals(strategy_name, epic, timeframe);

-- Trade outcome analysis
CREATE INDEX IF NOT EXISTS idx_backtest_signals_result ON backtest_signals(epic, trade_result, confidence_score DESC)
    WHERE trade_result IS NOT NULL;

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_backtest_signals_composite ON backtest_signals(
    epic, timeframe, strategy_name, signal_timestamp DESC
);

-- Performance lookup optimization
CREATE INDEX IF NOT EXISTS idx_backtest_performance_lookup ON backtest_performance(
    epic, timeframe, strategy_name, win_rate DESC NULLS LAST
);

-- Data quality monitoring
CREATE INDEX IF NOT EXISTS idx_backtest_signals_quality ON backtest_signals(data_completeness, validation_flags)
    WHERE data_completeness < 1.0;

-- Execution tracking
CREATE INDEX IF NOT EXISTS idx_backtest_executions_status ON backtest_executions(status, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_executions_strategy ON backtest_executions(strategy_name, data_start_date DESC);

-- Validation results
CREATE INDEX IF NOT EXISTS idx_backtest_signals_validation ON backtest_signals(validation_passed, confidence_score DESC);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to calculate performance metrics
CREATE OR REPLACE FUNCTION calculate_backtest_performance(p_execution_id INTEGER, p_epic VARCHAR DEFAULT NULL)
RETURNS VOID AS $$
DECLARE
    rec RECORD;
BEGIN
    -- Calculate performance for each epic/timeframe/strategy combination
    FOR rec IN
        SELECT epic, timeframe, strategy_name
        FROM backtest_signals
        WHERE execution_id = p_execution_id
          AND (p_epic IS NULL OR epic = p_epic)
        GROUP BY epic, timeframe, strategy_name
    LOOP
        INSERT INTO backtest_performance (
            execution_id, epic, timeframe, strategy_name,
            total_signals, bull_signals, bear_signals, validated_signals,
            winning_trades, losing_trades, breakeven_trades, win_rate,
            total_pips, avg_win_pips, avg_loss_pips, max_win_pips, max_loss_pips,
            profit_factor, expectancy_per_trade, max_drawdown_pips,
            avg_trade_duration_minutes, signal_frequency_per_day,
            data_completeness_score
        )
        SELECT
            p_execution_id,
            rec.epic,
            rec.timeframe,
            rec.strategy_name,
            COUNT(*) as total_signals,
            COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END) as bull_signals,
            COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END) as bear_signals,
            COUNT(CASE WHEN validation_passed = TRUE THEN 1 END) as validated_signals,
            COUNT(CASE WHEN trade_result = 'win' THEN 1 END) as winning_trades,
            COUNT(CASE WHEN trade_result = 'loss' THEN 1 END) as losing_trades,
            COUNT(CASE WHEN trade_result = 'breakeven' THEN 1 END) as breakeven_trades,
            CASE WHEN COUNT(CASE WHEN trade_result IN ('win', 'loss') THEN 1 END) > 0
                 THEN COUNT(CASE WHEN trade_result = 'win' THEN 1 END)::DECIMAL /
                      COUNT(CASE WHEN trade_result IN ('win', 'loss') THEN 1 END)
                 ELSE NULL END as win_rate,
            COALESCE(SUM(pips_gained), 0) as total_pips,
            AVG(CASE WHEN trade_result = 'win' THEN pips_gained END) as avg_win_pips,
            AVG(CASE WHEN trade_result = 'loss' THEN pips_gained END) as avg_loss_pips,
            MAX(CASE WHEN trade_result = 'win' THEN pips_gained END) as max_win_pips,
            MIN(CASE WHEN trade_result = 'loss' THEN pips_gained END) as max_loss_pips,
            CASE WHEN SUM(CASE WHEN trade_result = 'loss' THEN ABS(pips_gained) END) > 0
                 THEN SUM(CASE WHEN trade_result = 'win' THEN pips_gained ELSE 0 END) /
                      SUM(CASE WHEN trade_result = 'loss' THEN ABS(pips_gained) ELSE 0 END)
                 ELSE NULL END as profit_factor,
            AVG(pips_gained) as expectancy_per_trade,
            0 as max_drawdown_pips, -- TODO: Calculate running drawdown
            AVG(holding_time_minutes) as avg_trade_duration_minutes,
            COUNT(*)::DECIMAL / GREATEST(1, EXTRACT(days FROM MAX(signal_timestamp) - MIN(signal_timestamp))) as signal_frequency_per_day,
            AVG(data_completeness) as data_completeness_score
        FROM backtest_signals
        WHERE execution_id = p_execution_id
          AND epic = rec.epic
          AND timeframe = rec.timeframe
          AND strategy_name = rec.strategy_name
        ON CONFLICT (execution_id, epic, timeframe, strategy_name)
        DO UPDATE SET
            total_signals = EXCLUDED.total_signals,
            bull_signals = EXCLUDED.bull_signals,
            bear_signals = EXCLUDED.bear_signals,
            validated_signals = EXCLUDED.validated_signals,
            winning_trades = EXCLUDED.winning_trades,
            losing_trades = EXCLUDED.losing_trades,
            breakeven_trades = EXCLUDED.breakeven_trades,
            win_rate = EXCLUDED.win_rate,
            total_pips = EXCLUDED.total_pips,
            avg_win_pips = EXCLUDED.avg_win_pips,
            avg_loss_pips = EXCLUDED.avg_loss_pips,
            max_win_pips = EXCLUDED.max_win_pips,
            max_loss_pips = EXCLUDED.max_loss_pips,
            profit_factor = EXCLUDED.profit_factor,
            expectancy_per_trade = EXCLUDED.expectancy_per_trade,
            avg_trade_duration_minutes = EXCLUDED.avg_trade_duration_minutes,
            signal_frequency_per_day = EXCLUDED.signal_frequency_per_day,
            data_completeness_score = EXCLUDED.data_completeness_score,
            updated_at = NOW();
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get backtest summary
CREATE OR REPLACE FUNCTION get_backtest_summary(p_execution_id INTEGER)
RETURNS TABLE (
    execution_name VARCHAR,
    strategy VARCHAR,
    status VARCHAR,
    total_signals BIGINT,
    total_validated_signals BIGINT,
    avg_win_rate DECIMAL,
    total_pips DECIMAL,
    avg_profit_factor DECIMAL,
    data_quality DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        be.execution_name,
        be.strategy_name,
        be.status,
        COALESCE(SUM(bp.total_signals), 0) as total_signals,
        COALESCE(SUM(bp.validated_signals), 0) as total_validated_signals,
        AVG(bp.win_rate) as avg_win_rate,
        COALESCE(SUM(bp.total_pips), 0) as total_pips,
        AVG(bp.profit_factor) as avg_profit_factor,
        be.quality_score as data_quality
    FROM backtest_executions be
    LEFT JOIN backtest_performance bp ON be.id = bp.execution_id
    WHERE be.id = p_execution_id
    GROUP BY be.id, be.execution_name, be.strategy_name, be.status, be.quality_score;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Update backtest_executions.updated_at on changes
CREATE OR REPLACE FUNCTION update_backtest_executions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_backtest_executions_updated_at
    BEFORE UPDATE ON backtest_executions
    FOR EACH ROW EXECUTE FUNCTION update_backtest_executions_updated_at();

-- Update backtest_performance.updated_at on changes
CREATE OR REPLACE FUNCTION update_backtest_performance_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_backtest_performance_updated_at
    BEFORE UPDATE ON backtest_performance
    FOR EACH ROW EXECUTE FUNCTION update_backtest_performance_updated_at();

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- View for backtest signal analysis
CREATE OR REPLACE VIEW backtest_signals_analysis AS
SELECT
    bs.execution_id,
    be.execution_name,
    be.strategy_name,
    bs.epic,
    bs.timeframe,
    bs.signal_timestamp,
    bs.signal_type,
    bs.confidence_score,
    bs.entry_price,
    bs.exit_price,
    bs.pips_gained,
    bs.trade_result,
    bs.validation_passed,
    bs.data_completeness,
    be.data_start_date,
    be.data_end_date,
    be.status as execution_status
FROM backtest_signals bs
JOIN backtest_executions be ON bs.execution_id = be.id;

-- View for performance comparison
CREATE OR REPLACE VIEW backtest_performance_comparison AS
SELECT
    bp.execution_id,
    be.execution_name,
    bp.epic,
    bp.timeframe,
    bp.strategy_name,
    bp.total_signals,
    bp.validated_signals,
    bp.win_rate,
    bp.total_pips,
    bp.profit_factor,
    bp.expectancy_per_trade,
    bp.data_completeness_score,
    RANK() OVER (PARTITION BY bp.epic, bp.timeframe ORDER BY bp.profit_factor DESC NULLS LAST) as profit_factor_rank,
    RANK() OVER (PARTITION BY bp.epic, bp.timeframe ORDER BY bp.win_rate DESC NULLS LAST) as win_rate_rank
FROM backtest_performance bp
JOIN backtest_executions be ON bp.execution_id = be.id
WHERE be.status = 'completed';

-- Grant permissions (adjust as needed for your environment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_executions TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_signals TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_performance TO trading_user;
-- GRANT USAGE ON SEQUENCE backtest_executions_id_seq TO trading_user;
-- GRANT USAGE ON SEQUENCE backtest_signals_id_seq TO trading_user;
-- GRANT USAGE ON SEQUENCE backtest_performance_id_seq TO trading_user;