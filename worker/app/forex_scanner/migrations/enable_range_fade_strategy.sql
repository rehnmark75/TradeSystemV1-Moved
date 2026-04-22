-- Register RANGE_FADE (canonical name) in enabled_strategies.
-- Replaces the earlier RANGE_FADE_5M entry: the 15m profile has been retired,
-- so the strategy now emits the unsuffixed name "RANGE_FADE" only.
-- Run against: strategy_config database

INSERT INTO enabled_strategies (
    strategy_name,
    is_enabled,
    is_backtest_only,
    display_name,
    description,
    strategy_type
)
VALUES (
    'RANGE_FADE',
    TRUE,
    FALSE,
    'Range Fade',
    'Range fade (5m primary + 1h HTF) for mean-reverting setups',
    'signal'
)
ON CONFLICT (strategy_name) DO UPDATE
SET
    is_enabled = EXCLUDED.is_enabled,
    is_backtest_only = EXCLUDED.is_backtest_only,
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    strategy_type = EXCLUDED.strategy_type,
    updated_at = NOW();

-- Retire the 15m/5m suffixed entries so the scanner's selected-strategies set
-- doesn't carry stale keys. The strategy only emits "RANGE_FADE" now.
UPDATE enabled_strategies
SET is_enabled = FALSE, updated_at = NOW()
WHERE strategy_name IN ('RANGE_FADE_5M', 'RANGE_FADE_15M', 'EURUSD_RANGE_FADE', 'EURUSD_RANGE_FADE_5M');

-- Normalize any persisted strategy rows in the global config table to the
-- canonical "RANGE_FADE" name, so config loads don't re-seed the suffix.
UPDATE eurusd_range_fade_global_config
SET strategy_name = 'RANGE_FADE'
WHERE strategy_name IN ('RANGE_FADE_5M', 'RANGE_FADE_15M', 'EURUSD_RANGE_FADE', 'EURUSD_RANGE_FADE_5M');

-- Drop the retired 15m profile rows (5m is the only supported profile).
DELETE FROM eurusd_range_fade_global_config WHERE profile_name = '15m';
DELETE FROM eurusd_range_fade_pair_overrides WHERE profile_name = '15m';
