-- Migration: Add Signal Scanner Tables
-- Date: 2024-12-08
-- Description: Creates tables for the automated signal scanner system

-- ============================================================================
-- SCANNER DEFINITIONS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_signal_scanners (
    id SERIAL PRIMARY KEY,
    scanner_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    min_score_threshold INT DEFAULT 70,
    max_signals_per_run INT DEFAULT 50,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default scanners
INSERT INTO stock_signal_scanners (scanner_name, description, min_score_threshold) VALUES
    ('trend_momentum', 'Pullback entries in established uptrends with momentum confirmation', 70),
    ('breakout_confirmation', 'Volume-confirmed breakouts from consolidation patterns', 70),
    ('mean_reversion', 'Oversold bounces in uptrends with reversal patterns', 65),
    ('gap_and_go', 'Gap continuation plays with catalyst confirmation', 65)
ON CONFLICT (scanner_name) DO NOTHING;

-- ============================================================================
-- MAIN SIGNALS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_scanner_signals (
    id SERIAL PRIMARY KEY,
    signal_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scanner_name VARCHAR(100) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),

    -- Entry/Exit Levels
    entry_price DECIMAL(12, 4) NOT NULL,
    stop_loss DECIMAL(12, 4) NOT NULL,
    take_profit_1 DECIMAL(12, 4) NOT NULL,
    take_profit_2 DECIMAL(12, 4),
    risk_reward_ratio DECIMAL(5, 2),
    risk_percent DECIMAL(5, 2),  -- % risk from entry to stop

    -- Scoring (0-100 scale)
    composite_score INT NOT NULL CHECK (composite_score >= 0 AND composite_score <= 100),
    quality_tier VARCHAR(5) NOT NULL CHECK (quality_tier IN ('A+', 'A', 'B', 'C', 'D')),
    trend_score DECIMAL(5, 2),
    momentum_score DECIMAL(5, 2),
    volume_score DECIMAL(5, 2),
    pattern_score DECIMAL(5, 2),
    confluence_score DECIMAL(5, 2),

    -- Context
    setup_description TEXT,
    confluence_factors TEXT[],
    timeframe VARCHAR(10) DEFAULT 'daily',
    market_regime VARCHAR(50),

    -- Position Sizing
    suggested_position_size_pct DECIMAL(5, 2),
    max_risk_per_trade_pct DECIMAL(5, 2) DEFAULT 1.5,

    -- Status Tracking
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'triggered', 'partial_exit', 'closed', 'expired', 'cancelled')),
    trigger_timestamp TIMESTAMPTZ,
    close_timestamp TIMESTAMPTZ,
    close_price DECIMAL(12, 4),
    realized_pnl_pct DECIMAL(8, 2),
    realized_r_multiple DECIMAL(5, 2),
    exit_reason VARCHAR(100),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT fk_scanner FOREIGN KEY (scanner_name) REFERENCES stock_signal_scanners(scanner_name),
    CONSTRAINT valid_stop_loss CHECK (
        (signal_type = 'BUY' AND stop_loss < entry_price) OR
        (signal_type = 'SELL' AND stop_loss > entry_price)
    ),
    CONSTRAINT valid_take_profit CHECK (
        (signal_type = 'BUY' AND take_profit_1 > entry_price) OR
        (signal_type = 'SELL' AND take_profit_1 < entry_price)
    )
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_scanner_signals_ticker ON stock_scanner_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_scanner_signals_timestamp ON stock_scanner_signals(signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scanner_signals_status ON stock_scanner_signals(status);
CREATE INDEX IF NOT EXISTS idx_scanner_signals_quality ON stock_scanner_signals(quality_tier, composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_scanner_signals_scanner ON stock_scanner_signals(scanner_name, signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scanner_signals_active ON stock_scanner_signals(status, quality_tier) WHERE status = 'active';

-- ============================================================================
-- PERFORMANCE TRACKING TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_scanner_performance (
    id SERIAL PRIMARY KEY,
    scanner_name VARCHAR(100) NOT NULL,
    evaluation_date DATE NOT NULL,
    evaluation_period VARCHAR(20) DEFAULT 'daily',  -- 'daily', 'weekly', 'monthly'

    -- Signal Counts
    total_signals INT DEFAULT 0,
    signals_triggered INT DEFAULT 0,
    signals_closed INT DEFAULT 0,
    signals_expired INT DEFAULT 0,

    -- Performance Metrics
    win_rate DECIMAL(5, 2),
    avg_win_pct DECIMAL(6, 2),
    avg_loss_pct DECIMAL(6, 2),
    profit_factor DECIMAL(6, 2),
    avg_r_multiple DECIMAL(5, 2),
    max_r_multiple DECIMAL(5, 2),
    min_r_multiple DECIMAL(5, 2),
    expectancy DECIMAL(6, 2),

    -- Quality Tier Distribution
    a_plus_signals INT DEFAULT 0,
    a_plus_win_rate DECIMAL(5, 2),
    a_signals INT DEFAULT 0,
    a_win_rate DECIMAL(5, 2),
    b_signals INT DEFAULT 0,
    b_win_rate DECIMAL(5, 2),

    -- Time Analysis
    avg_hold_time_hours DECIMAL(8, 2),
    avg_time_to_trigger_hours DECIMAL(8, 2),

    -- Risk Metrics
    max_drawdown_pct DECIMAL(6, 2),
    sharpe_ratio DECIMAL(5, 2),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_perf_scanner FOREIGN KEY (scanner_name) REFERENCES stock_signal_scanners(scanner_name),
    CONSTRAINT unique_scanner_date UNIQUE (scanner_name, evaluation_date, evaluation_period)
);

CREATE INDEX IF NOT EXISTS idx_scanner_perf_date ON stock_scanner_performance(evaluation_date DESC);
CREATE INDEX IF NOT EXISTS idx_scanner_perf_scanner ON stock_scanner_performance(scanner_name, evaluation_date DESC);

-- ============================================================================
-- SIGNAL HISTORY / AUDIT TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_scanner_signal_history (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL,
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,  -- 'created', 'triggered', 'tp1_hit', 'tp2_hit', 'stop_hit', 'expired', 'cancelled'
    previous_status VARCHAR(20),
    new_status VARCHAR(20),
    price_at_event DECIMAL(12, 4),
    notes TEXT,

    CONSTRAINT fk_signal FOREIGN KEY (signal_id) REFERENCES stock_scanner_signals(id)
);

CREATE INDEX IF NOT EXISTS idx_signal_history_signal ON stock_scanner_signal_history(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_history_timestamp ON stock_scanner_signal_history(event_timestamp DESC);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update timestamp on modification
CREATE OR REPLACE FUNCTION update_scanner_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for auto-updating timestamps
DROP TRIGGER IF EXISTS update_scanner_signals_timestamp ON stock_scanner_signals;
CREATE TRIGGER update_scanner_signals_timestamp
    BEFORE UPDATE ON stock_scanner_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_scanner_timestamp();

DROP TRIGGER IF EXISTS update_scanners_timestamp ON stock_signal_scanners;
CREATE TRIGGER update_scanners_timestamp
    BEFORE UPDATE ON stock_signal_scanners
    FOR EACH ROW
    EXECUTE FUNCTION update_scanner_timestamp();

-- Function to log signal status changes
CREATE OR REPLACE FUNCTION log_signal_status_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO stock_scanner_signal_history (
            signal_id, event_type, previous_status, new_status, price_at_event
        ) VALUES (
            NEW.id,
            CASE
                WHEN NEW.status = 'triggered' THEN 'triggered'
                WHEN NEW.status = 'closed' THEN
                    CASE
                        WHEN NEW.realized_pnl_pct > 0 THEN 'profit_close'
                        ELSE 'stop_hit'
                    END
                WHEN NEW.status = 'expired' THEN 'expired'
                WHEN NEW.status = 'cancelled' THEN 'cancelled'
                ELSE 'status_change'
            END,
            OLD.status,
            NEW.status,
            NEW.close_price
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS log_signal_changes ON stock_scanner_signals;
CREATE TRIGGER log_signal_changes
    AFTER UPDATE ON stock_scanner_signals
    FOR EACH ROW
    EXECUTE FUNCTION log_signal_status_change();

-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

-- Active signals view with full details
CREATE OR REPLACE VIEW v_active_scanner_signals AS
SELECT
    s.*,
    i.name as company_name,
    w.tier,
    w.score as watchlist_score,
    w.relative_volume,
    w.atr_percent
FROM stock_scanner_signals s
JOIN stock_instruments i ON s.ticker = i.ticker
LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
    AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
WHERE s.status = 'active'
ORDER BY s.composite_score DESC;

-- Scanner performance summary view
CREATE OR REPLACE VIEW v_scanner_performance_summary AS
SELECT
    scanner_name,
    COUNT(*) as total_days_tracked,
    ROUND(AVG(win_rate), 2) as avg_win_rate,
    ROUND(AVG(profit_factor), 2) as avg_profit_factor,
    ROUND(AVG(avg_r_multiple), 2) as avg_r_multiple,
    SUM(total_signals) as total_signals_all_time,
    SUM(signals_triggered) as total_triggered,
    SUM(signals_closed) as total_closed,
    MAX(evaluation_date) as last_evaluation
FROM stock_scanner_performance
GROUP BY scanner_name
ORDER BY avg_profit_factor DESC;

-- Today's signals by quality
CREATE OR REPLACE VIEW v_todays_signals AS
SELECT
    quality_tier,
    COUNT(*) as signal_count,
    ROUND(AVG(composite_score), 1) as avg_score,
    ROUND(AVG(risk_reward_ratio), 2) as avg_rr,
    array_agg(DISTINCT ticker ORDER BY ticker) as tickers
FROM stock_scanner_signals
WHERE DATE(signal_timestamp) = CURRENT_DATE
    AND status = 'active'
GROUP BY quality_tier
ORDER BY quality_tier;

-- Comments
COMMENT ON TABLE stock_scanner_signals IS 'Stores all signals generated by the automated scanner system';
COMMENT ON TABLE stock_scanner_performance IS 'Daily/weekly/monthly performance tracking for each scanner';
COMMENT ON TABLE stock_signal_scanners IS 'Configuration for each scanner type';
COMMENT ON VIEW v_active_scanner_signals IS 'Active signals with company and watchlist details';
