-- Backtest Trades Table
-- Creates a dedicated table for storing historical backtest trade decisions
-- Captures the complete decision-making process of the trading pipeline on historical data
-- Created: 2025-09-25

-- Drop table if exists (for testing/development - uncomment if needed)
-- DROP TABLE IF EXISTS backtest_trades;

-- Main backtest trades table
CREATE TABLE IF NOT EXISTS backtest_trades (
    id SERIAL PRIMARY KEY,

    -- Backtest metadata
    backtest_session_id VARCHAR(64), -- Identifier for the backtest run
    historical_timestamp TIMESTAMP NOT NULL, -- The historical timestamp being analyzed
    decision_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, -- When the decision was made

    -- Signal information
    epic VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- BULL, BEAR, UNKNOWN
    timeframe VARCHAR(10) NOT NULL DEFAULT '15m',

    -- Trading decision
    trade_action VARCHAR(10) NOT NULL, -- BUY, SELL, REJECT, ERROR
    decision_reason TEXT, -- Why the decision was made
    pipeline_stage VARCHAR(20) DEFAULT 'COMPLETE', -- Stage where processing completed

    -- Price and entry information
    entry_price DECIMAL(10,5),
    market_price DECIMAL(10,5),
    spread_pips DECIMAL(6,3),

    -- Confidence and scoring
    confidence_score DECIMAL(6,4) NOT NULL DEFAULT 0,
    intelligence_score DECIMAL(6,4) DEFAULT 0,
    intelligence_threshold DECIMAL(6,4) DEFAULT 0,

    -- Claude analysis results (if available)
    claude_enabled BOOLEAN DEFAULT FALSE,
    claude_score DECIMAL(6,4) DEFAULT 0,
    claude_reasoning TEXT,
    claude_recommendation VARCHAR(10), -- BUY, SELL, REJECT, HOLD
    claude_result JSON, -- Full Claude analysis result

    -- Signal validation results
    validation_passed BOOLEAN DEFAULT TRUE,
    validation_error TEXT,
    validation_details JSON, -- Detailed validation results

    -- Risk management results
    risk_evaluation BOOLEAN DEFAULT TRUE,
    risk_details JSON,

    -- Original signal data
    signal_data JSON NOT NULL, -- Complete original signal
    validated_signal_data JSON, -- Signal after validation

    -- Market context (if available)
    market_regime VARCHAR(30), -- trending, ranging, breakout, reversal
    market_session VARCHAR(20), -- asian, london, new_york, overlap
    volatility_level VARCHAR(20), -- low, medium, high, very_high
    market_intelligence JSON, -- Market intelligence context

    -- Technical indicators (key ones for analysis)
    ema_alignment VARCHAR(20), -- bullish, bearish, neutral, mixed
    macd_signal VARCHAR(20), -- bullish, bearish, neutral
    trend_strength DECIMAL(6,4),
    momentum_score DECIMAL(6,4),

    -- Performance tracking (for future analysis)
    alert_id INTEGER, -- Reference to alert_history if saved there too
    execution_status VARCHAR(20) DEFAULT 'SIMULATED', -- SIMULATED, EXECUTED, FAILED

    -- Timestamps for tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_backtest_trades_session ON backtest_trades(backtest_session_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_historical_time ON backtest_trades(historical_timestamp);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_epic_strategy ON backtest_trades(epic, strategy);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_action ON backtest_trades(trade_action);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_confidence ON backtest_trades(confidence_score);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_claude ON backtest_trades(claude_enabled, claude_recommendation);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_timeframe ON backtest_trades(timeframe);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_backtest_trades_epic_time ON backtest_trades(epic, historical_timestamp);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_session_action ON backtest_trades(backtest_session_id, trade_action);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_strategy_action ON backtest_trades(strategy, trade_action);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_backtest_trades_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_backtest_trades_updated_at
    BEFORE UPDATE ON backtest_trades
    FOR EACH ROW
    EXECUTE FUNCTION update_backtest_trades_updated_at();

-- Add comments for documentation
COMMENT ON TABLE backtest_trades IS 'Stores historical backtest trade decisions with complete pipeline context';
COMMENT ON COLUMN backtest_trades.backtest_session_id IS 'Identifier for the backtest run session';
COMMENT ON COLUMN backtest_trades.historical_timestamp IS 'The historical timestamp being analyzed (when the signal would have occurred)';
COMMENT ON COLUMN backtest_trades.decision_timestamp IS 'When the backtest decision was made (processing timestamp)';
COMMENT ON COLUMN backtest_trades.trade_action IS 'Final trade decision: BUY, SELL, REJECT, ERROR';
COMMENT ON COLUMN backtest_trades.decision_reason IS 'Explanation of why this decision was made';
COMMENT ON COLUMN backtest_trades.intelligence_score IS 'Orchestrator intelligence filtering score';
COMMENT ON COLUMN backtest_trades.claude_result IS 'Complete Claude analysis result (JSON)';
COMMENT ON COLUMN backtest_trades.signal_data IS 'Original signal data from scanner (JSON)';
COMMENT ON COLUMN backtest_trades.validated_signal_data IS 'Signal after validation processing (JSON)';

-- Create views for common analysis queries
CREATE OR REPLACE VIEW backtest_trade_summary AS
SELECT
    backtest_session_id,
    epic,
    strategy,
    timeframe,
    COUNT(*) as total_decisions,
    COUNT(CASE WHEN trade_action = 'BUY' THEN 1 END) as buy_decisions,
    COUNT(CASE WHEN trade_action = 'SELL' THEN 1 END) as sell_decisions,
    COUNT(CASE WHEN trade_action = 'REJECT' THEN 1 END) as reject_decisions,
    COUNT(CASE WHEN trade_action = 'ERROR' THEN 1 END) as error_decisions,
    AVG(confidence_score) as avg_confidence,
    AVG(intelligence_score) as avg_intelligence_score,
    AVG(CASE WHEN claude_enabled THEN claude_score END) as avg_claude_score,
    MIN(historical_timestamp) as first_signal,
    MAX(historical_timestamp) as last_signal
FROM backtest_trades
GROUP BY backtest_session_id, epic, strategy, timeframe;

COMMENT ON VIEW backtest_trade_summary IS 'Summary view of backtest trade decisions by session, epic, and strategy';

-- Create view for decision analysis
CREATE OR REPLACE VIEW backtest_decision_analysis AS
SELECT
    DATE_TRUNC('hour', historical_timestamp) as signal_hour,
    epic,
    strategy,
    trade_action,
    market_session,
    market_regime,
    COUNT(*) as decision_count,
    AVG(confidence_score) as avg_confidence,
    AVG(intelligence_score) as avg_intelligence
FROM backtest_trades
WHERE trade_action IN ('BUY', 'SELL', 'REJECT')
GROUP BY DATE_TRUNC('hour', historical_timestamp), epic, strategy, trade_action, market_session, market_regime
ORDER BY signal_hour DESC;

COMMENT ON VIEW backtest_decision_analysis IS 'Analysis view for backtest decisions by time, market conditions, and strategy';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON backtest_trades TO forex_scanner_user;
-- GRANT SELECT ON backtest_trade_summary TO forex_scanner_user;
-- GRANT SELECT ON backtest_decision_analysis TO forex_scanner_user;

-- Success message
SELECT 'Backtest trades table and supporting structures created successfully!' as status;