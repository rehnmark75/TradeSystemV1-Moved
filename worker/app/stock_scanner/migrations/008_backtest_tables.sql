-- Migration 008: Stock Backtest Tables
-- Creates tables for storing backtest execution results and signals

-- =============================================================================
-- BACKTEST EXECUTIONS TABLE
-- =============================================================================
-- Stores metadata for each backtest run

CREATE TABLE IF NOT EXISTS stock_backtest_executions (
    id SERIAL PRIMARY KEY,
    execution_name VARCHAR(255),
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    tickers TEXT[],                         -- Array of tickers tested
    timeframe VARCHAR(10) DEFAULT '1d',
    config_snapshot JSONB,                  -- Strategy parameters used

    -- Execution status
    status VARCHAR(20) DEFAULT 'running',   -- 'running', 'completed', 'failed'
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,

    -- Summary statistics
    total_signals INTEGER DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winners INTEGER DEFAULT 0,
    losers INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    total_pnl_percent DECIMAL(10,4),
    avg_win_percent DECIMAL(10,4),
    avg_loss_percent DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    max_drawdown_percent DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),

    -- Additional metadata
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_backtest_exec_strategy
    ON stock_backtest_executions(strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_exec_status
    ON stock_backtest_executions(status);
CREATE INDEX IF NOT EXISTS idx_backtest_exec_created
    ON stock_backtest_executions(created_at DESC);

-- =============================================================================
-- BACKTEST SIGNALS TABLE
-- =============================================================================
-- Stores individual signals and their simulated outcomes

CREATE TABLE IF NOT EXISTS stock_backtest_signals (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES stock_backtest_executions(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    signal_timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(10) NOT NULL,       -- 'BUY' or 'SELL'

    -- Entry levels
    entry_price DECIMAL(12,4) NOT NULL,
    stop_loss_price DECIMAL(12,4),
    take_profit_price DECIMAL(12,4),
    risk_reward_ratio DECIMAL(6,2),

    -- Signal quality
    confidence DECIMAL(5,4),
    quality_tier VARCHAR(5),                -- 'A+', 'A', 'B', 'C', 'D'

    -- Trade outcome (from simulation)
    exit_price DECIMAL(12,4),
    exit_timestamp TIMESTAMPTZ,
    exit_reason VARCHAR(20),                -- 'TP_HIT', 'SL_HIT', 'TIMEOUT', 'MANUAL'
    trade_result VARCHAR(10),               -- 'WIN', 'LOSS', 'BREAKEVEN'
    pnl_percent DECIMAL(10,4),
    pnl_amount DECIMAL(12,4),
    holding_days INTEGER,

    -- Technical context at signal time
    ema_20 DECIMAL(12,4),
    ema_50 DECIMAL(12,4),
    ema_100 DECIMAL(12,4),
    ema_200 DECIMAL(12,4),
    rsi DECIMAL(6,2),
    atr DECIMAL(12,4),
    pullback_percent DECIMAL(6,4),          -- How far below 20 EMA at entry

    -- Stock context
    sector VARCHAR(100),
    volume BIGINT,
    relative_volume DECIMAL(6,2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_backtest_sig_exec
    ON stock_backtest_signals(execution_id);
CREATE INDEX IF NOT EXISTS idx_backtest_sig_ticker
    ON stock_backtest_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_backtest_sig_result
    ON stock_backtest_signals(trade_result);
CREATE INDEX IF NOT EXISTS idx_backtest_sig_timestamp
    ON stock_backtest_signals(signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_sig_quality
    ON stock_backtest_signals(quality_tier, confidence DESC);

-- =============================================================================
-- PERFORMANCE ANALYTICS VIEW
-- =============================================================================
-- Aggregated view for easy performance analysis

CREATE OR REPLACE VIEW v_backtest_performance AS
SELECT
    e.id,
    e.execution_name,
    e.strategy_name,
    e.start_date,
    e.end_date,
    e.timeframe,
    e.status,
    e.started_at,
    e.completed_at,
    e.duration_seconds,

    -- Execution stats
    e.total_signals,
    e.total_trades,
    e.winners,
    e.losers,
    e.win_rate,
    e.total_pnl_percent,
    e.avg_win_percent,
    e.avg_loss_percent,
    e.profit_factor,
    e.max_drawdown_percent,
    e.sharpe_ratio,

    -- Aggregated from signals
    COUNT(s.id) as actual_signal_count,
    COUNT(DISTINCT s.ticker) as unique_tickers,
    AVG(s.pnl_percent) FILTER (WHERE s.trade_result = 'WIN') as calc_avg_win,
    AVG(s.pnl_percent) FILTER (WHERE s.trade_result = 'LOSS') as calc_avg_loss,
    AVG(s.holding_days) as avg_holding_days,
    AVG(s.confidence) as avg_confidence,

    -- Breakdown by exit reason
    COUNT(*) FILTER (WHERE s.exit_reason = 'TP_HIT') as tp_hit_count,
    COUNT(*) FILTER (WHERE s.exit_reason = 'SL_HIT') as sl_hit_count,
    COUNT(*) FILTER (WHERE s.exit_reason = 'TIMEOUT') as timeout_count,

    -- Quality tier breakdown
    COUNT(*) FILTER (WHERE s.quality_tier = 'A+') as aplus_signals,
    COUNT(*) FILTER (WHERE s.quality_tier = 'A') as a_signals,
    COUNT(*) FILTER (WHERE s.quality_tier = 'B') as b_signals,
    COUNT(*) FILTER (WHERE s.quality_tier = 'C') as c_signals,

    -- Tickers with signals
    array_agg(DISTINCT s.ticker) FILTER (WHERE s.ticker IS NOT NULL) as tickers_with_signals

FROM stock_backtest_executions e
LEFT JOIN stock_backtest_signals s ON e.id = s.execution_id
GROUP BY e.id;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get the latest backtest for a strategy
CREATE OR REPLACE FUNCTION get_latest_backtest(p_strategy VARCHAR)
RETURNS TABLE (
    id INTEGER,
    execution_name VARCHAR,
    start_date DATE,
    end_date DATE,
    win_rate DECIMAL,
    total_pnl_percent DECIMAL,
    profit_factor DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.execution_name,
        e.start_date,
        e.end_date,
        e.win_rate,
        e.total_pnl_percent,
        e.profit_factor
    FROM stock_backtest_executions e
    WHERE e.strategy_name = p_strategy
      AND e.status = 'completed'
    ORDER BY e.created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to compare strategy performance
CREATE OR REPLACE FUNCTION compare_strategies(p_days INTEGER DEFAULT 90)
RETURNS TABLE (
    strategy_name VARCHAR,
    backtest_count BIGINT,
    avg_win_rate DECIMAL,
    avg_pnl DECIMAL,
    avg_profit_factor DECIMAL,
    best_pnl DECIMAL,
    worst_pnl DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.strategy_name,
        COUNT(*)::BIGINT as backtest_count,
        AVG(e.win_rate)::DECIMAL(5,2) as avg_win_rate,
        AVG(e.total_pnl_percent)::DECIMAL(10,4) as avg_pnl,
        AVG(e.profit_factor)::DECIMAL(10,4) as avg_profit_factor,
        MAX(e.total_pnl_percent)::DECIMAL(10,4) as best_pnl,
        MIN(e.total_pnl_percent)::DECIMAL(10,4) as worst_pnl
    FROM stock_backtest_executions e
    WHERE e.status = 'completed'
      AND e.created_at > NOW() - (p_days || ' days')::INTERVAL
    GROUP BY e.strategy_name
    ORDER BY avg_pnl DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE stock_backtest_executions IS 'Stores metadata and summary statistics for each backtest run';
COMMENT ON TABLE stock_backtest_signals IS 'Stores individual signals and their simulated trade outcomes';
COMMENT ON VIEW v_backtest_performance IS 'Aggregated view for easy backtest performance analysis';
COMMENT ON FUNCTION get_latest_backtest IS 'Returns the most recent completed backtest for a given strategy';
COMMENT ON FUNCTION compare_strategies IS 'Compares performance metrics across different strategies';
