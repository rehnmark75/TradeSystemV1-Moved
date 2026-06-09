-- ============================================================================
-- SMC_MOMENTUM hardening - June 2026
--
-- Forward evidence showed weak basket performance and bad risk geometry leaking
-- to order validation. Keep broad cells in monitoring while only allowing the
-- best observed cell (NZDUSD SELL) to trade in demo.
-- ============================================================================

INSERT INTO smc_momentum_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
VALUES
    ('demo', 'version', '1.2.0', 'string', 'general', 'SMC_MOMENTUM hardening: direction-correct risk, final confidence, rejection candle quality'),
    ('demo', 'min_reclaim_pct_of_range', '0.15', 'float', 'filters', 'Minimum close-back-inside distance as fraction of sweep candle range'),
    ('demo', 'min_body_pct_of_range', '0.20', 'float', 'filters', 'Minimum sweep candle body as fraction of candle range')
ON CONFLICT (config_set, parameter_name) DO UPDATE
SET parameter_value = EXCLUDED.parameter_value,
    value_type = EXCLUDED.value_type,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    updated_at = NOW();

INSERT INTO smc_momentum_global_config
    (config_set, parameter_name, parameter_value, value_type, category, description)
VALUES
    ('live', 'version', '1.2.0', 'string', 'general', 'SMC_MOMENTUM hardening: live remains non-trading'),
    ('live', 'min_reclaim_pct_of_range', '0.15', 'float', 'filters', 'Minimum close-back-inside distance as fraction of sweep candle range'),
    ('live', 'min_body_pct_of_range', '0.20', 'float', 'filters', 'Minimum sweep candle body as fraction of candle range')
ON CONFLICT (config_set, parameter_name) DO UPDATE
SET parameter_value = EXCLUDED.parameter_value,
    value_type = EXCLUDED.value_type,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    updated_at = NOW();

-- Demo: keep scanning the basket, but only NZDUSD SELL remains order-eligible.
UPDATE smc_momentum_pair_overrides
SET is_enabled = TRUE,
    is_traded = FALSE,
    monitor_only = TRUE,
    parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'monitor_only', true,
            'min_reclaim_pct_of_range', 0.15,
            'min_body_pct_of_range', 0.20
        ),
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026 hardening:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026 hardening: monitor-only until forward outcomes prove pair/direction edge.'
    END
WHERE config_set = 'demo';

UPDATE smc_momentum_pair_overrides
SET is_enabled = TRUE,
    is_traded = TRUE,
    monitor_only = FALSE,
    parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'monitor_only', false,
            'allowed_directions', jsonb_build_array('SELL'),
            'blocked_hours_utc', jsonb_build_array(17, 18, 19, 20),
            'min_reclaim_pct_of_range', 0.15,
            'min_body_pct_of_range', 0.20
        ),
    updated_at = NOW(),
    notes = CASE
        WHEN COALESCE(notes, '') LIKE '%Jun 2026 hardening:%' THEN notes
        ELSE COALESCE(notes || E'\n', '') || 'Jun 2026 hardening: only NZDUSD SELL remains demo-traded; all other SMC_MOMENTUM cells monitor-only.'
    END
WHERE config_set = 'demo'
  AND epic = 'CS.D.NZDUSD.MINI.IP';

-- Live: explicit non-trading posture.
UPDATE smc_momentum_pair_overrides
SET is_enabled = FALSE,
    is_traded = FALSE,
    monitor_only = TRUE,
    parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object('monitor_only', true),
    updated_at = NOW()
WHERE config_set = 'live';

SELECT
    config_set,
    pair_name,
    is_enabled,
    is_traded,
    monitor_only,
    parameter_overrides
FROM smc_momentum_pair_overrides
ORDER BY config_set, pair_name;
