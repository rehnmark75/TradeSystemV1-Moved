-- Migration 019: Add TradingView-style technical indicator columns
-- Adds oscillators and moving averages for TradingView summary gauges

-- Add new oscillator columns to stock_screening_metrics
ALTER TABLE stock_screening_metrics

-- Stochastic Oscillator (14, 3, 3)
ADD COLUMN IF NOT EXISTS stoch_k DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS stoch_d DECIMAL(6,2),

-- CCI (Commodity Channel Index, 20 period)
ADD COLUMN IF NOT EXISTS cci_20 DECIMAL(8,2),

-- ADX (Average Directional Index, 14 period) - may already exist, adding conditionally
ADD COLUMN IF NOT EXISTS adx_14 DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS plus_di DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS minus_di DECIMAL(6,2),

-- Awesome Oscillator (5, 34)
ADD COLUMN IF NOT EXISTS ao_value DECIMAL(12,4),

-- Momentum (10 period)
ADD COLUMN IF NOT EXISTS momentum_10 DECIMAL(8,2),

-- Stochastic RSI (3, 3, 14, 14)
ADD COLUMN IF NOT EXISTS stoch_rsi_k DECIMAL(6,2),
ADD COLUMN IF NOT EXISTS stoch_rsi_d DECIMAL(6,2),

-- Williams %R (14 period)
ADD COLUMN IF NOT EXISTS williams_r DECIMAL(6,2),

-- Bull Bear Power (13 EMA)
ADD COLUMN IF NOT EXISTS bull_power DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS bear_power DECIMAL(12,4),

-- Ultimate Oscillator (7, 14, 28)
ADD COLUMN IF NOT EXISTS ultimate_osc DECIMAL(6,2),

-- Additional Moving Averages
ADD COLUMN IF NOT EXISTS ema_10 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS ema_30 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS ema_50 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS ema_100 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS ema_200 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS sma_10 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS sma_30 DECIMAL(12,4),
ADD COLUMN IF NOT EXISTS sma_100 DECIMAL(12,4),

-- Ichimoku Base Line (Kijun-sen: 26 period)
ADD COLUMN IF NOT EXISTS ichimoku_base DECIMAL(12,4),

-- Volume Weighted Moving Average (20 period)
ADD COLUMN IF NOT EXISTS vwma_20 DECIMAL(12,4),

-- Pre-calculated TradingView summary counts (for fast queries)
ADD COLUMN IF NOT EXISTS tv_osc_buy INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS tv_osc_sell INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS tv_osc_neutral INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS tv_ma_buy INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS tv_ma_sell INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS tv_ma_neutral INTEGER DEFAULT 0,

-- Overall summary signal
ADD COLUMN IF NOT EXISTS tv_overall_signal VARCHAR(15),  -- STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL
ADD COLUMN IF NOT EXISTS tv_overall_score DECIMAL(5,2);  -- -100 to +100 scale

-- Add index on tv_overall_signal for filtering
CREATE INDEX IF NOT EXISTS idx_tv_overall_signal ON stock_screening_metrics(tv_overall_signal);

-- Add composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_tv_summary ON stock_screening_metrics(ticker, calculation_date, tv_overall_signal);

-- Add comments for documentation
COMMENT ON COLUMN stock_screening_metrics.stoch_k IS 'Stochastic %K (14,3,3) - Fast stochastic oscillator';
COMMENT ON COLUMN stock_screening_metrics.stoch_d IS 'Stochastic %D (14,3,3) - Slow stochastic signal line';
COMMENT ON COLUMN stock_screening_metrics.cci_20 IS 'Commodity Channel Index (20 period)';
COMMENT ON COLUMN stock_screening_metrics.adx_14 IS 'Average Directional Index (14 period) - Trend strength';
COMMENT ON COLUMN stock_screening_metrics.plus_di IS 'Positive Directional Indicator (+DI)';
COMMENT ON COLUMN stock_screening_metrics.minus_di IS 'Negative Directional Indicator (-DI)';
COMMENT ON COLUMN stock_screening_metrics.ao_value IS 'Awesome Oscillator (5,34) - Momentum indicator';
COMMENT ON COLUMN stock_screening_metrics.momentum_10 IS 'Momentum (10 period) - Rate of price change';
COMMENT ON COLUMN stock_screening_metrics.stoch_rsi_k IS 'Stochastic RSI %K (3,3,14,14)';
COMMENT ON COLUMN stock_screening_metrics.stoch_rsi_d IS 'Stochastic RSI %D (3,3,14,14)';
COMMENT ON COLUMN stock_screening_metrics.williams_r IS 'Williams %R (14 period) - Momentum oscillator';
COMMENT ON COLUMN stock_screening_metrics.bull_power IS 'Bull Power - High minus 13 EMA';
COMMENT ON COLUMN stock_screening_metrics.bear_power IS 'Bear Power - Low minus 13 EMA';
COMMENT ON COLUMN stock_screening_metrics.ultimate_osc IS 'Ultimate Oscillator (7,14,28) - Multi-timeframe momentum';
COMMENT ON COLUMN stock_screening_metrics.ichimoku_base IS 'Ichimoku Base Line (Kijun-sen, 26 period)';
COMMENT ON COLUMN stock_screening_metrics.vwma_20 IS 'Volume Weighted Moving Average (20 period)';
COMMENT ON COLUMN stock_screening_metrics.tv_osc_buy IS 'Count of oscillators signaling BUY';
COMMENT ON COLUMN stock_screening_metrics.tv_osc_sell IS 'Count of oscillators signaling SELL';
COMMENT ON COLUMN stock_screening_metrics.tv_osc_neutral IS 'Count of oscillators signaling NEUTRAL';
COMMENT ON COLUMN stock_screening_metrics.tv_ma_buy IS 'Count of moving averages signaling BUY (price > MA)';
COMMENT ON COLUMN stock_screening_metrics.tv_ma_sell IS 'Count of moving averages signaling SELL (price < MA)';
COMMENT ON COLUMN stock_screening_metrics.tv_ma_neutral IS 'Count of moving averages signaling NEUTRAL (price near MA)';
COMMENT ON COLUMN stock_screening_metrics.tv_overall_signal IS 'TradingView overall summary: STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL';
COMMENT ON COLUMN stock_screening_metrics.tv_overall_score IS 'TradingView overall score (-100 to +100, calculated from all indicators)';
