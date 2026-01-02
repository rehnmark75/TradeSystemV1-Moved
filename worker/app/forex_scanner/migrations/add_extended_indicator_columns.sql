-- Migration: Add Extended Indicator Columns for Trade Analysis
-- Date: 2026-01-02
-- Purpose: Add KAMA, Bollinger Bands, Stochastic, Supertrend, and other indicators
--          to enable comprehensive analysis of winning vs losing trades
--
-- Run with: docker exec postgres psql -U postgres -d forex -f /path/to/this/file.sql

BEGIN;

-- ============================================================================
-- KAMA Indicator Columns
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS kama_value DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS kama_er DECIMAL(6,4);  -- Efficiency Ratio 0-1
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS kama_trend VARCHAR(10);  -- up/down/neutral
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS kama_signal VARCHAR(20);  -- crossover signal

-- ============================================================================
-- Bollinger Bands Columns
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS bb_upper DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS bb_middle DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS bb_lower DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS bb_width DECIMAL(10,6);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS bb_percent_b DECIMAL(6,4);  -- %B indicator (0-1, can exceed)
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS price_vs_bb VARCHAR(20);  -- above_upper/in_band/below_lower

-- ============================================================================
-- Stochastic Oscillator Columns
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS stoch_k DECIMAL(6,2);  -- %K line
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS stoch_d DECIMAL(6,2);  -- %D line (signal)
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS stoch_zone VARCHAR(20);  -- overbought/neutral/oversold

-- ============================================================================
-- Supertrend Columns
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS supertrend_value DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS supertrend_direction INTEGER;  -- 1=bullish, -1=bearish

-- ============================================================================
-- Additional RSI Details
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS rsi_zone VARCHAR(20);  -- overbought/neutral/oversold
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS rsi_divergence VARCHAR(20);  -- bullish/bearish/none

-- ============================================================================
-- EMA Relationships
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS ema_9 DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS ema_21 DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS ema_50 DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS ema_200 DECIMAL(10,5);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS price_vs_ema_200 VARCHAR(10);  -- above/below
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS ema_stack_order VARCHAR(30);  -- bullish/bearish/mixed

-- ============================================================================
-- Candle Pattern Info
-- ============================================================================
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS candle_body_pips DECIMAL(8,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS candle_upper_wick_pips DECIMAL(8,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS candle_lower_wick_pips DECIMAL(8,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS candle_type VARCHAR(20);  -- bullish/bearish/doji

-- ============================================================================
-- Add same columns to smc_simple_rejections for rejection analysis
-- ============================================================================
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS kama_value DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS kama_er DECIMAL(6,4);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS kama_trend VARCHAR(10);

ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS bb_upper DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS bb_middle DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS bb_lower DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS bb_width DECIMAL(10,6);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS bb_percent_b DECIMAL(6,4);

ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS stoch_k DECIMAL(6,2);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS stoch_d DECIMAL(6,2);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS stoch_zone VARCHAR(20);

ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS supertrend_value DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS supertrend_direction INTEGER;

ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS rsi_zone VARCHAR(20);

ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS ema_9 DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS ema_21 DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS ema_50 DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS ema_200 DECIMAL(10,5);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS price_vs_ema_200 VARCHAR(10);

-- ============================================================================
-- Create Indexes for Analysis Queries
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_alert_kama_er ON alert_history(kama_er);
CREATE INDEX IF NOT EXISTS idx_alert_bb_percent_b ON alert_history(bb_percent_b);
CREATE INDEX IF NOT EXISTS idx_alert_stoch_zone ON alert_history(stoch_zone);
CREATE INDEX IF NOT EXISTS idx_alert_rsi_zone ON alert_history(rsi_zone);
CREATE INDEX IF NOT EXISTS idx_alert_supertrend_dir ON alert_history(supertrend_direction);
CREATE INDEX IF NOT EXISTS idx_alert_ema_stack ON alert_history(ema_stack_order);

CREATE INDEX IF NOT EXISTS idx_rejection_kama_er ON smc_simple_rejections(kama_er);
CREATE INDEX IF NOT EXISTS idx_rejection_stoch_zone ON smc_simple_rejections(stoch_zone);
CREATE INDEX IF NOT EXISTS idx_rejection_supertrend ON smc_simple_rejections(supertrend_direction);

COMMIT;

-- ============================================================================
-- Verification Query
-- ============================================================================
SELECT 'alert_history new columns:' as info;
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'alert_history'
AND column_name IN ('kama_value', 'kama_er', 'bb_upper', 'bb_percent_b', 'stoch_k', 'supertrend_value', 'ema_200')
ORDER BY column_name;

SELECT 'smc_simple_rejections new columns:' as info;
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'smc_simple_rejections'
AND column_name IN ('kama_value', 'kama_er', 'bb_upper', 'stoch_k', 'supertrend_value')
ORDER BY column_name;
