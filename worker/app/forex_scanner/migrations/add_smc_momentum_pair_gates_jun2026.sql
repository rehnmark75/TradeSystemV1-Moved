-- ============================================================================
-- SMC_MOMENTUM pair quality gates - June 2026
--
-- Adds pair-level direction/session gates discovered from historical
-- SMC_MOMENTUM signal analysis. Demo config only; live remains unchanged until
-- promoted intentionally.
-- ============================================================================

UPDATE smc_momentum_global_config
SET parameter_value = '1.1.0',
    updated_at = NOW()
WHERE config_set = 'demo'
  AND parameter_name = 'version';

UPDATE smc_momentum_pair_overrides
SET parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) || '{"allowed_directions":["SELL"],"blocked_hours_utc":[17,18,19,20]}'::jsonb,
    monitor_only = FALSE,
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026: BEAR-only plus 17-20 UTC block from SMC_MOMENTUM profitability analysis.'
    END
WHERE config_set = 'demo'
  AND epic = 'CS.D.AUDUSD.MINI.IP';

UPDATE smc_momentum_pair_overrides
SET parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) || '{"allowed_directions":["SELL"],"blocked_hours_utc":[17,18,19,20]}'::jsonb,
    is_enabled = TRUE,
    monitor_only = FALSE,
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026: BEAR-only plus 17-20 UTC block from SMC_MOMENTUM profitability analysis.'
    END
WHERE config_set = 'demo'
  AND epic = 'CS.D.NZDUSD.MINI.IP';

UPDATE smc_momentum_pair_overrides
SET parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) || '{"allowed_directions":["BUY"],"blocked_hours_utc":[17,18,19,20]}'::jsonb,
    monitor_only = FALSE,
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026: BULL-only plus 17-20 UTC block from SMC_MOMENTUM profitability analysis.'
    END
WHERE config_set = 'demo'
  AND epic = 'CS.D.EURJPY.MINI.IP';

UPDATE smc_momentum_pair_overrides
SET parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) || '{"allowed_directions":["BUY"],"blocked_hours_utc":[17,18,19,20]}'::jsonb,
    monitor_only = FALSE,
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026: BULL-only plus 17-20 UTC block from SMC_MOMENTUM profitability analysis.'
    END
WHERE config_set = 'demo'
  AND epic = 'CS.D.AUDJPY.MINI.IP';

UPDATE smc_momentum_pair_overrides
SET parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) || '{"allowed_directions":["BUY"],"allowed_hours_utc":[0,1,2,9,15,16,17,19]}'::jsonb,
    is_enabled = TRUE,
    is_traded = FALSE,
    monitor_only = FALSE,
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026: enable monitor with BULL-only selected-hour rule from SMC_MOMENTUM profitability analysis.'
    END
WHERE config_set = 'demo'
  AND epic = 'CS.D.USDJPY.MINI.IP';

SELECT config_set, epic, pair_name, is_enabled, is_traded, monitor_only, parameter_overrides
FROM smc_momentum_pair_overrides
WHERE config_set = 'demo'
  AND epic IN (
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.AUDJPY.MINI.IP',
    'CS.D.USDJPY.MINI.IP'
  )
ORDER BY pair_name;
