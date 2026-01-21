-- ============================================================================
-- Add Scalp Reversal Override Config (Counter-Trend)
-- Database: strategy_config
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/add_scalp_reversal_override_config.sql
-- ============================================================================

\c strategy_config;

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS scalp_reversal_enabled BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS scalp_reversal_min_runway_pips DECIMAL(6,2) DEFAULT 15.0,
    ADD COLUMN IF NOT EXISTS scalp_reversal_min_entry_momentum DECIMAL(4,2) DEFAULT 0.60,
    ADD COLUMN IF NOT EXISTS scalp_reversal_block_regimes TEXT[] DEFAULT ARRAY['breakout'],
    ADD COLUMN IF NOT EXISTS scalp_reversal_block_volatility_states TEXT[] DEFAULT ARRAY['high'],
    ADD COLUMN IF NOT EXISTS scalp_reversal_allow_rsi_extremes BOOLEAN DEFAULT TRUE;

UPDATE smc_simple_global_config
SET
    scalp_reversal_enabled = COALESCE(scalp_reversal_enabled, TRUE),
    scalp_reversal_min_runway_pips = COALESCE(scalp_reversal_min_runway_pips, 15.0),
    scalp_reversal_min_entry_momentum = COALESCE(scalp_reversal_min_entry_momentum, 0.60),
    scalp_reversal_block_regimes = COALESCE(scalp_reversal_block_regimes, ARRAY['breakout']),
    scalp_reversal_block_volatility_states = COALESCE(scalp_reversal_block_volatility_states, ARRAY['high']),
    scalp_reversal_allow_rsi_extremes = COALESCE(scalp_reversal_allow_rsi_extremes, TRUE)
WHERE scalp_reversal_enabled IS NULL
   OR scalp_reversal_min_runway_pips IS NULL
   OR scalp_reversal_min_entry_momentum IS NULL
   OR scalp_reversal_block_regimes IS NULL
   OR scalp_reversal_block_volatility_states IS NULL
   OR scalp_reversal_allow_rsi_extremes IS NULL;

COMMENT ON COLUMN smc_simple_global_config.scalp_reversal_enabled IS
    'Enable counter-trend reversal override when HTF alignment fails';
COMMENT ON COLUMN smc_simple_global_config.scalp_reversal_min_runway_pips IS
    'Minimum runway in pips to opposing S/R for reversal scalps';
COMMENT ON COLUMN smc_simple_global_config.scalp_reversal_min_entry_momentum IS
    'Minimum entry candle momentum (0-1) for reversal override';
COMMENT ON COLUMN smc_simple_global_config.scalp_reversal_block_regimes IS
    'Market regimes that block reversal override (e.g., breakout)';
COMMENT ON COLUMN smc_simple_global_config.scalp_reversal_block_volatility_states IS
    'Volatility states that block reversal override (e.g., high)';
COMMENT ON COLUMN smc_simple_global_config.scalp_reversal_allow_rsi_extremes IS
    'Allow RSI overbought/oversold to satisfy reversal confirmation';
