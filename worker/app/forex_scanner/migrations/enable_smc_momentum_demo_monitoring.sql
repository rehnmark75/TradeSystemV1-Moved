-- Enable SMC_MOMENTUM through the demo processing pipeline without allowing orders.
-- The strategy's own pair overrides still decide which pairs detect signals, and
-- monitor_only remains true unless a pair is explicitly promoted.

UPDATE scanner_global_config
SET enabled_strategies = (
        SELECT jsonb_agg(strategy_name)
        FROM (
            SELECT DISTINCT strategy_name
            FROM jsonb_array_elements_text(enabled_strategies || '["SMC_MOMENTUM"]'::jsonb) AS t(strategy_name)
            ORDER BY strategy_name
        ) s
    ),
    claude_vision_strategies = (
        SELECT jsonb_agg(strategy_name)
        FROM (
            SELECT DISTINCT strategy_name
            FROM jsonb_array_elements_text(claude_vision_strategies || '["SMC_MOMENTUM"]'::jsonb) AS t(strategy_name)
            ORDER BY strategy_name
        ) s
    ),
    updated_at = NOW(),
    updated_by = 'codex',
    change_reason = 'Enable SMC_MOMENTUM demo monitoring pipeline; keep orders blocked by monitor_only pair config'
WHERE config_set = 'demo'
  AND is_active = TRUE;

INSERT INTO enabled_strategies (
    strategy_name,
    is_enabled,
    is_backtest_only,
    display_name,
    description,
    strategy_type
)
VALUES (
    'SMC_MOMENTUM',
    TRUE,
    FALSE,
    'SMC Momentum',
    'Liquidity sweep plus rejection wick monitor strategy',
    'signal'
)
ON CONFLICT (strategy_name) DO UPDATE
SET is_enabled = TRUE,
    is_backtest_only = FALSE,
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    strategy_type = EXCLUDED.strategy_type,
    updated_at = NOW();
