-- Migration: Add Codex Filter Columns to smc_simple_pair_overrides
-- Date: 2026-01-29
-- Purpose: Enable per-pair Codex-recommended filters (EMA stack, ranging block, min ADX, etc.)
-- Analysis: USDCHF trade analysis shows these filters would improve performance

-- Add per-pair Codex filter columns
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_require_ema_stack_alignment BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_block_ranging_market BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_block_low_volatility_trending BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_min_adx NUMERIC(5,2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS scalp_block_breakout_regime BOOLEAN DEFAULT NULL;

-- Add comments explaining each filter
COMMENT ON COLUMN smc_simple_pair_overrides.scalp_require_ema_stack_alignment IS
    'Require EMA stack to match trade direction (BUY=bullish stack, SELL=bearish). NULL=use global config.';

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_block_ranging_market IS
    'Block trades in ranging market regime. Codex analysis: 0% WR in ranging for some pairs. NULL=use global config.';

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_block_low_volatility_trending IS
    'Block trades in trending+low_volatility conditions. Codex: 20% WR in this regime. NULL=use global config.';

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_min_adx IS
    'Minimum ADX threshold to take trades. Codex: 90% of USDCAD losses had ADX < 20. NULL=use global config.';

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_block_breakout_regime IS
    'Block trades in breakout regime. Codex: 0% WR in breakout for some pairs. NULL=use global config.';

-- Verify columns were added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'smc_simple_pair_overrides'
AND column_name IN (
    'scalp_require_ema_stack_alignment',
    'scalp_block_ranging_market',
    'scalp_block_low_volatility_trending',
    'scalp_min_adx',
    'scalp_block_breakout_regime'
)
ORDER BY column_name;
