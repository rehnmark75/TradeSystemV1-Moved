-- Migration: 012_premarket_signals.sql
-- Description: Create table for pre-market signals from Finnhub
-- Date: 2024-12-29

-- Pre-market signals table
-- Stores signals generated from pre-market quotes and overnight news
CREATE TABLE IF NOT EXISTS stock_premarket_signals (
    id SERIAL PRIMARY KEY,

    -- Signal identification
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,  -- GAP_PLAY, MOMENTUM, NEWS_CATALYST, REVERSAL
    direction VARCHAR(10) NOT NULL,     -- BUY or SELL
    strength VARCHAR(20) NOT NULL,      -- STRONG, MODERATE, WEAK
    confidence DECIMAL(4,3) NOT NULL,   -- 0.0 to 1.0

    -- Gap data
    gap_percent DECIMAL(8,3),           -- Gap % from previous close
    gap_type VARCHAR(30),               -- gap_up_large, gap_down_small, etc.
    current_price DECIMAL(12,4),        -- Pre-market price
    previous_close DECIMAL(12,4),       -- Previous day close

    -- News context
    news_count INT DEFAULT 0,
    news_sentiment_score DECIMAL(5,3),  -- -1.0 to 1.0
    news_sentiment_level VARCHAR(20),   -- very_bullish, bullish, neutral, bearish, very_bearish
    key_headlines TEXT[],               -- Top headlines array

    -- Entry/Exit levels
    suggested_entry DECIMAL(12,4),
    suggested_stop DECIMAL(12,4),
    suggested_target DECIMAL(12,4),
    risk_reward DECIMAL(6,3),

    -- Timestamps
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Outcome tracking (filled in after market close)
    outcome_status VARCHAR(20),          -- triggered, hit_target, hit_stop, expired
    actual_entry_price DECIMAL(12,4),
    actual_exit_price DECIMAL(12,4),
    realized_pnl_pct DECIMAL(8,3),
    outcome_recorded_at TIMESTAMPTZ
);

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS idx_premarket_signals_generated
    ON stock_premarket_signals(generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_premarket_signals_symbol
    ON stock_premarket_signals(symbol);

CREATE INDEX IF NOT EXISTS idx_premarket_signals_confidence
    ON stock_premarket_signals(confidence DESC);

CREATE INDEX IF NOT EXISTS idx_premarket_signals_type
    ON stock_premarket_signals(signal_type, direction);

-- Pipeline log entry type for premarket pricing
-- (uses existing stock_pipeline_log table)

-- Add comment for documentation
COMMENT ON TABLE stock_premarket_signals IS
    'Pre-market signals generated from Finnhub quotes and overnight news. '
    'Runs at 9:00 AM ET (30 min before market open). '
    'Detects gaps, analyzes overnight news sentiment, and generates signals.';

COMMENT ON COLUMN stock_premarket_signals.gap_type IS
    'Gap classification: gap_up_large (>5%), gap_up_medium (2-5%), '
    'gap_up_small (0.5-2%), flat, gap_down_small, gap_down_medium, gap_down_large';

COMMENT ON COLUMN stock_premarket_signals.signal_type IS
    'Signal type: GAP_PLAY (pure gap), NEWS_CATALYST (gap + supporting news), '
    'REVERSAL (gap contradicted by news), MOMENTUM (continuation)';
