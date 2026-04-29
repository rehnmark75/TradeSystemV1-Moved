-- Per-strategy trailing config (Apr 2026)
--
-- Adds a `strategy` dimension to `trailing_pair_config` so each strategy can
-- have its own per-pair trailing tuning. Existing rows become the
-- 'DEFAULT' strategy and continue to apply to any strategy that has no
-- dedicated row (fallback chain handled in TrailingConfigService).
--
-- Idempotent: safe to re-run.

BEGIN;

-- 1. Column
ALTER TABLE trailing_pair_config
    ADD COLUMN IF NOT EXISTS strategy text NOT NULL DEFAULT 'DEFAULT';

-- 2. Allowed values
ALTER TABLE trailing_pair_config
    DROP CONSTRAINT IF EXISTS trailing_pair_config_strategy_check;
ALTER TABLE trailing_pair_config
    ADD CONSTRAINT trailing_pair_config_strategy_check
    CHECK (strategy IN (
        'DEFAULT',
        'SMC_SIMPLE',
        'XAU_GOLD',
        'RANGE_FADE',
        'MEAN_REVERSION',
        'RANGE_STRUCTURE'
    ));

-- 3. Replace unique scope to include strategy
ALTER TABLE trailing_pair_config
    DROP CONSTRAINT IF EXISTS trailing_unique_scope;
ALTER TABLE trailing_pair_config
    ADD CONSTRAINT trailing_unique_scope
    UNIQUE (strategy, config_set, epic, is_scalp);

-- 4. Indexes
DROP INDEX IF EXISTS idx_trailing_lookup;
CREATE INDEX IF NOT EXISTS idx_trailing_lookup_strategy
    ON trailing_pair_config (strategy, config_set, epic, is_scalp)
    WHERE is_active = true;

COMMIT;
