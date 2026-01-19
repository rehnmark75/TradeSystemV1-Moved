-- Migration: 018_trading_metrics_columns.sql
-- Description: Add trading metrics columns to stock_watchlist_results for trade planning
-- Date: 2026-01-18

-- ATR and volatility metrics
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS atr_14 DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS atr_percent DECIMAL(6, 2);

-- Support/Resistance levels from SMC analysis
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS swing_high DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS swing_low DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS swing_high_date DATE;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS swing_low_date DATE;

-- Nearest order block (key level)
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS nearest_ob_price DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS nearest_ob_type VARCHAR(10);  -- 'bullish' or 'bearish'
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS nearest_ob_distance DECIMAL(6, 2);  -- % distance from current price

-- Trade plan metrics (auto-calculated)
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS suggested_entry_low DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS suggested_entry_high DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS suggested_stop_loss DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS suggested_target_1 DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS suggested_target_2 DECIMAL(10, 4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS risk_reward_ratio DECIMAL(4, 2);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS risk_percent DECIMAL(6, 2);  -- % risk from entry to stop

-- Relative strength context
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS rs_percentile INTEGER;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS rs_trend VARCHAR(20);  -- 'improving', 'stable', 'deteriorating'
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS rs_5d_change INTEGER;  -- change in RS percentile over 5 days

-- Sector context
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS sector VARCHAR(100);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS sector_rank INTEGER;  -- rank within sector
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS sector_total INTEGER;  -- total stocks in sector

-- Earnings and catalyst timing
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS earnings_date DATE;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS days_to_earnings INTEGER;

-- Volume context
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS volume_trend VARCHAR(20);  -- 'accumulation', 'distribution', 'neutral'
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS relative_volume DECIMAL(6, 2);  -- vs 20-day avg

-- 5-day average daily change (already exists but ensure it's there)
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS avg_daily_change_5d DECIMAL(6, 2);

-- Comments
COMMENT ON COLUMN stock_watchlist_results.atr_percent IS 'ATR as percentage of price - key for position sizing';
COMMENT ON COLUMN stock_watchlist_results.swing_high IS 'Recent swing high from SMC analysis - resistance level';
COMMENT ON COLUMN stock_watchlist_results.swing_low IS 'Recent swing low from SMC analysis - support level';
COMMENT ON COLUMN stock_watchlist_results.suggested_stop_loss IS 'Auto-calculated stop loss based on ATR and support';
COMMENT ON COLUMN stock_watchlist_results.suggested_target_1 IS 'First profit target (typically 1:1.5 or 1:2 R:R)';
COMMENT ON COLUMN stock_watchlist_results.risk_reward_ratio IS 'Risk to reward ratio for suggested trade plan';
COMMENT ON COLUMN stock_watchlist_results.rs_5d_change IS 'Change in RS percentile over last 5 days (positive = improving)';
COMMENT ON COLUMN stock_watchlist_results.volume_trend IS 'Recent volume pattern: accumulation, distribution, or neutral';

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_watchlist_results_rs ON stock_watchlist_results(rs_percentile) WHERE rs_percentile IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_watchlist_results_earnings ON stock_watchlist_results(days_to_earnings) WHERE days_to_earnings IS NOT NULL AND days_to_earnings <= 14;
