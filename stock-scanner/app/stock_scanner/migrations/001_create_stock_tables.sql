-- =============================================================================
-- STOCK SCANNER DATABASE SCHEMA
-- Migration: 001_create_stock_tables.sql
-- Description: Create database tables for stock scanner
-- Database: stocks (separate from forex)
-- =============================================================================

-- =============================================================================
-- STOCK INSTRUMENTS (synced from RoboMarkets)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_instruments (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(200),
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(20),  -- NYSE, NASDAQ, LSE, etc.
    currency VARCHAR(10) DEFAULT 'USD',

    -- RoboMarkets specific fields
    robomarkets_ticker VARCHAR(50),  -- Internal RoboMarkets ticker format
    contract_size DECIMAL(10,4) DEFAULT 1,
    min_quantity DECIMAL(10,4),
    max_quantity DECIMAL(10,4),
    quantity_step DECIMAL(10,6),

    -- Trading status
    is_active BOOLEAN DEFAULT TRUE,
    is_tradeable BOOLEAN DEFAULT TRUE,

    -- Metadata
    market_cap VARCHAR(20),  -- 'large', 'mid', 'small', 'micro'
    avg_volume BIGINT,

    -- Sync tracking
    last_sync TIMESTAMP DEFAULT NOW(),
    metadata JSONB,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- STOCK CANDLES (OHLCV data from yfinance)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_candles (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- '1h', '4h', '1d'
    timestamp TIMESTAMP NOT NULL,

    -- OHLCV data
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT,

    -- Adjusted prices (for splits/dividends)
    adjusted_close DECIMAL(12,4),

    -- Data quality
    is_complete BOOLEAN DEFAULT TRUE,  -- False for current candle
    source VARCHAR(20) DEFAULT 'yfinance',

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_stock_candle UNIQUE(ticker, timeframe, timestamp)
);

-- =============================================================================
-- SYNTHESIZED CANDLES (4H, Daily from 1H base)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_candles_synthesized (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- '4h', '1d' (synthesized from 1h)
    timestamp TIMESTAMP NOT NULL,

    -- OHLCV data
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT,

    -- Synthesis metadata
    source_timeframe VARCHAR(10) DEFAULT '1h',
    candles_used INTEGER,  -- Number of source candles used

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_synthesized_candle UNIQUE(ticker, timeframe, timestamp)
);

-- =============================================================================
-- STOCK SIGNALS
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_signals (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_timestamp TIMESTAMP NOT NULL,

    -- Signal details
    signal_type VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL'
    strategy_name VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,4),
    signal_strength DECIMAL(5,4),

    -- Price levels
    entry_price DECIMAL(12,4) NOT NULL,
    stop_loss_price DECIMAL(12,4),
    take_profit_price DECIMAL(12,4),
    stop_loss_percent DECIMAL(5,2),
    take_profit_percent DECIMAL(5,2),
    risk_reward_ratio DECIMAL(6,3),

    -- Technical indicators snapshot
    indicator_values JSONB,

    -- Stock-specific context
    sector VARCHAR(100),
    market_cap VARCHAR(20),
    relative_volume DECIMAL(6,2),  -- Volume vs average
    gap_percent DECIMAL(6,2),  -- Gap from previous close

    -- Validation
    validation_passed BOOLEAN DEFAULT FALSE,
    validation_reasons TEXT[],

    -- Higher timeframe context
    daily_trend VARCHAR(10),  -- 'bullish', 'bearish', 'neutral'
    weekly_trend VARCHAR(10),

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_stock_signal_type CHECK (signal_type IN ('BUY', 'SELL'))
);

-- =============================================================================
-- STOCK ALERTS (for notifications)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_alerts (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    signal_id BIGINT REFERENCES stock_signals(id),

    -- Alert details
    alert_type VARCHAR(20) NOT NULL,  -- 'signal', 'order_filled', 'stop_hit', etc.
    signal_type VARCHAR(10),
    strategy_name VARCHAR(50),
    confidence_score DECIMAL(5,4),
    price DECIMAL(12,4),

    -- Message
    message TEXT,

    -- Status
    sent BOOLEAN DEFAULT FALSE,
    sent_at TIMESTAMP,
    executed BOOLEAN DEFAULT FALSE,
    executed_at TIMESTAMP,

    -- Deduplication
    alert_hash VARCHAR(64),

    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- STOCK ORDERS (tracking orders sent to RoboMarkets)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_orders (
    id BIGSERIAL PRIMARY KEY,
    signal_id BIGINT REFERENCES stock_signals(id),

    -- Order identification
    robomarkets_order_id VARCHAR(100),
    ticker VARCHAR(20) NOT NULL,

    -- Order details
    order_type VARCHAR(20) NOT NULL,  -- 'market', 'limit', 'stop'
    side VARCHAR(10) NOT NULL,  -- 'buy', 'sell'
    quantity DECIMAL(12,4) NOT NULL,
    price DECIMAL(12,4),  -- For limit orders

    -- Stop/Take profit
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),

    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'submitted', 'filled', 'cancelled', 'rejected'
    filled_price DECIMAL(12,4),
    filled_quantity DECIMAL(12,4),
    filled_at TIMESTAMP,

    -- Error tracking
    error_message TEXT,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_order_status CHECK (status IN ('pending', 'submitted', 'filled', 'partially_filled', 'cancelled', 'rejected'))
);

-- =============================================================================
-- STOCK POSITIONS (open positions)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_positions (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT REFERENCES stock_orders(id),
    robomarkets_deal_id VARCHAR(100),

    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'long', 'short'
    quantity DECIMAL(12,4) NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,

    -- Current status
    current_price DECIMAL(12,4),
    unrealized_pnl DECIMAL(12,4),
    unrealized_pnl_percent DECIMAL(6,2),

    -- Risk management
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),

    -- Position status
    status VARCHAR(20) DEFAULT 'open',  -- 'open', 'closed'
    exit_price DECIMAL(12,4),
    exit_reason VARCHAR(30),  -- 'take_profit', 'stop_loss', 'manual', 'signal'
    realized_pnl DECIMAL(12,4),

    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- DATA SYNC LOG (tracking yfinance fetches)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stock_data_sync_log (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Sync details
    sync_type VARCHAR(20),  -- 'full', 'incremental'
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    candles_fetched INTEGER,
    candles_inserted INTEGER,
    candles_updated INTEGER,

    -- Status
    status VARCHAR(20),  -- 'success', 'partial', 'failed'
    error_message TEXT,

    -- Performance
    duration_seconds DECIMAL(8,2),

    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Candle queries (most frequent)
CREATE INDEX IF NOT EXISTS idx_stock_candles_ticker_tf_time
    ON stock_candles(ticker, timeframe, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_stock_candles_timestamp
    ON stock_candles(timestamp DESC);

-- Synthesized candles
CREATE INDEX IF NOT EXISTS idx_stock_candles_synth_ticker_tf_time
    ON stock_candles_synthesized(ticker, timeframe, timestamp DESC);

-- Signals
CREATE INDEX IF NOT EXISTS idx_stock_signals_ticker_time
    ON stock_signals(ticker, signal_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_stock_signals_strategy
    ON stock_signals(strategy_name, signal_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_stock_signals_validation
    ON stock_signals(validation_passed, confidence_score DESC);

-- Instruments
CREATE INDEX IF NOT EXISTS idx_stock_instruments_active
    ON stock_instruments(is_active, is_tradeable);

CREATE INDEX IF NOT EXISTS idx_stock_instruments_sector
    ON stock_instruments(sector) WHERE is_active = TRUE;

-- Orders and positions
CREATE INDEX IF NOT EXISTS idx_stock_orders_status
    ON stock_orders(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_stock_positions_open
    ON stock_positions(status, ticker) WHERE status = 'open';

-- Alerts
CREATE INDEX IF NOT EXISTS idx_stock_alerts_pending
    ON stock_alerts(sent, created_at DESC) WHERE sent = FALSE;

-- Sync log
CREATE INDEX IF NOT EXISTS idx_stock_sync_log_ticker
    ON stock_data_sync_log(ticker, timeframe, created_at DESC);

-- =============================================================================
-- TRIGGERS FOR UPDATED_AT
-- =============================================================================

CREATE OR REPLACE FUNCTION update_stock_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_stock_instruments_updated_at
    BEFORE UPDATE ON stock_instruments
    FOR EACH ROW EXECUTE FUNCTION update_stock_updated_at();

CREATE TRIGGER trigger_stock_orders_updated_at
    BEFORE UPDATE ON stock_orders
    FOR EACH ROW EXECUTE FUNCTION update_stock_updated_at();

CREATE TRIGGER trigger_stock_positions_updated_at
    BEFORE UPDATE ON stock_positions
    FOR EACH ROW EXECUTE FUNCTION update_stock_updated_at();

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to synthesize higher timeframe candles from 1h data
CREATE OR REPLACE FUNCTION synthesize_candles(
    p_ticker VARCHAR,
    p_target_timeframe VARCHAR,  -- '4h' or '1d'
    p_start_date TIMESTAMP DEFAULT NULL,
    p_end_date TIMESTAMP DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    v_interval INTERVAL;
    v_count INTEGER := 0;
BEGIN
    -- Determine interval
    v_interval := CASE p_target_timeframe
        WHEN '4h' THEN INTERVAL '4 hours'
        WHEN '1d' THEN INTERVAL '1 day'
        ELSE INTERVAL '4 hours'
    END;

    -- Insert synthesized candles
    INSERT INTO stock_candles_synthesized (
        ticker, timeframe, timestamp,
        open, high, low, close, volume,
        source_timeframe, candles_used
    )
    SELECT
        ticker,
        p_target_timeframe,
        date_trunc('hour', timestamp) -
            (EXTRACT(hour FROM timestamp)::INTEGER %
                CASE p_target_timeframe WHEN '4h' THEN 4 ELSE 24 END) * INTERVAL '1 hour',
        FIRST_VALUE(open) OVER w,
        MAX(high) OVER w,
        MIN(low) OVER w,
        LAST_VALUE(close) OVER w,
        SUM(volume) OVER w,
        '1h',
        COUNT(*) OVER w
    FROM stock_candles
    WHERE ticker = p_ticker
      AND timeframe = '1h'
      AND (p_start_date IS NULL OR timestamp >= p_start_date)
      AND (p_end_date IS NULL OR timestamp <= p_end_date)
    WINDOW w AS (
        PARTITION BY ticker,
            date_trunc('hour', timestamp) -
                (EXTRACT(hour FROM timestamp)::INTEGER %
                    CASE p_target_timeframe WHEN '4h' THEN 4 ELSE 24 END) * INTERVAL '1 hour'
        ORDER BY timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )
    ON CONFLICT (ticker, timeframe, timestamp)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        candles_used = EXCLUDED.candles_used;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest candles for a ticker
CREATE OR REPLACE FUNCTION get_latest_candles(
    p_ticker VARCHAR,
    p_timeframe VARCHAR DEFAULT '1h',
    p_limit INTEGER DEFAULT 200
) RETURNS TABLE (
    timestamp TIMESTAMP,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.timestamp,
        c.open,
        c.high,
        c.low,
        c.close,
        c.volume
    FROM stock_candles c
    WHERE c.ticker = p_ticker
      AND c.timeframe = p_timeframe
    ORDER BY c.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View for active instruments with latest price
CREATE OR REPLACE VIEW stock_instruments_summary AS
SELECT
    i.ticker,
    i.name,
    i.sector,
    i.exchange,
    i.market_cap,
    i.is_tradeable,
    c.close as latest_price,
    c.volume as latest_volume,
    c.timestamp as price_timestamp
FROM stock_instruments i
LEFT JOIN LATERAL (
    SELECT close, volume, timestamp
    FROM stock_candles
    WHERE ticker = i.ticker AND timeframe = '1h'
    ORDER BY timestamp DESC
    LIMIT 1
) c ON TRUE
WHERE i.is_active = TRUE;

-- View for recent signals
CREATE OR REPLACE VIEW stock_signals_recent AS
SELECT
    s.*,
    i.name as stock_name,
    i.exchange
FROM stock_signals s
JOIN stock_instruments i ON s.ticker = i.ticker
WHERE s.signal_timestamp > NOW() - INTERVAL '7 days'
ORDER BY s.signal_timestamp DESC;

-- =============================================================================
-- GRANTS (adjust as needed)
-- =============================================================================
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO trading_user;

COMMENT ON TABLE stock_instruments IS 'Stock instruments synced from RoboMarkets API';
COMMENT ON TABLE stock_candles IS 'Historical OHLCV data from yfinance (1h primary)';
COMMENT ON TABLE stock_candles_synthesized IS 'Higher timeframe candles synthesized from 1h data';
COMMENT ON TABLE stock_signals IS 'Trading signals generated by strategies';
COMMENT ON TABLE stock_orders IS 'Orders sent to RoboMarkets';
COMMENT ON TABLE stock_positions IS 'Open and closed positions';
