-- Migration: Add ADX and Technical Indicators to alert_history
-- Purpose: Store ADX, RSI, ATR values and validation details for trade analysis
-- Date: 2025-10-13
-- Safe: Uses IF NOT EXISTS to prevent breaking existing data

-- Add technical indicator columns
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS adx NUMERIC(6,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS adx_plus NUMERIC(6,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS adx_minus NUMERIC(6,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS rsi NUMERIC(6,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS atr NUMERIC(10,6);

-- Add validation and trigger metadata
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS trigger_type VARCHAR(20);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS validation_details JSON;
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS swing_proximity_distance NUMERIC(8,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS swing_proximity_valid BOOLEAN;

-- Add market bias tracking
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS market_bias VARCHAR(20);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS market_bias_conflict BOOLEAN;
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS directional_consensus NUMERIC(4,3);

-- Create indexes for query performance
CREATE INDEX IF NOT EXISTS idx_alert_history_adx ON alert_history(adx);
CREATE INDEX IF NOT EXISTS idx_alert_history_rsi ON alert_history(rsi);
CREATE INDEX IF NOT EXISTS idx_alert_history_trigger_type ON alert_history(trigger_type);
CREATE INDEX IF NOT EXISTS idx_alert_history_market_bias ON alert_history(market_bias);
CREATE INDEX IF NOT EXISTS idx_alert_history_bias_conflict ON alert_history(market_bias_conflict);
CREATE INDEX IF NOT EXISTS idx_alert_history_swing_valid ON alert_history(swing_proximity_valid);

-- Add comments for documentation
COMMENT ON COLUMN alert_history.adx IS 'Average Directional Index - trend strength indicator (0-100)';
COMMENT ON COLUMN alert_history.adx_plus IS 'Positive Directional Indicator (+DI)';
COMMENT ON COLUMN alert_history.adx_minus IS 'Negative Directional Indicator (-DI)';
COMMENT ON COLUMN alert_history.rsi IS 'Relative Strength Index (0-100)';
COMMENT ON COLUMN alert_history.atr IS 'Average True Range - volatility measure';
COMMENT ON COLUMN alert_history.trigger_type IS 'Signal trigger type: macd, adx, ema_crossover, etc.';
COMMENT ON COLUMN alert_history.validation_details IS 'JSON containing full validation context and reasons';
COMMENT ON COLUMN alert_history.swing_proximity_distance IS 'Distance to nearest swing point in pips';
COMMENT ON COLUMN alert_history.swing_proximity_valid IS 'Whether signal passes swing proximity validation';
COMMENT ON COLUMN alert_history.market_bias IS 'Overall market bias: bullish, bearish, neutral';
COMMENT ON COLUMN alert_history.market_bias_conflict IS 'True if signal direction conflicts with market bias';
COMMENT ON COLUMN alert_history.directional_consensus IS 'Market directional consensus score (0-1)';

-- Migration complete
SELECT 'ADX and technical indicators migration completed successfully' AS status;
