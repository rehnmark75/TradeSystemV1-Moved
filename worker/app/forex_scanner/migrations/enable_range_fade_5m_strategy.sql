-- Enable RANGE_FADE_5M in the multi-strategy scanner configuration.
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
    'RANGE_FADE_5M',
    TRUE,
    FALSE,
    'Range Fade 5m',
    'Range fade strategy for mean-reverting setups on the 5m profile',
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
