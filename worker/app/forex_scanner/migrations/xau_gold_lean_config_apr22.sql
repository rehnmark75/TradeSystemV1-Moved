-- ============================================================================
-- XAU_GOLD lean launch config (Apr 22 2026)
-- ============================================================================
-- 90d inverse-ablation (scripts/ablate_xau_gold.py, scripts/ablate_xau_followups.sh)
-- identified two edge-lifting gates:
--   1. require_ob_or_fvg=true                         -> +0.25 PF
--   2. block_ranging=true + adx_trending_threshold=25 -> +1.48 PF on top of (1)
--
-- Shipped v1.0 collapsed to 4 signals over 90d (PF 0.00) due to gate-stacking.
-- This lean config keeps the two real gates, drops the sample-size killers, and
-- flips pair to monitor_only=true for 2-4w forward observation.
--
-- Projected (90d BT): ~23 signals/month, PF 3.83, WR 65.7%, exp +38.9 pips.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. GLOBAL CONFIG — apply to BOTH config_sets so demo/live stay in sync
-- ---------------------------------------------------------------------------

-- Edge-lifting gates (KEEP ON)
UPDATE xau_gold_global_config SET parameter_value = 'true',
    change_reason = 'apr22 ablation: +0.25 PF on top of permissive'
    WHERE parameter_name = 'require_ob_or_fvg';

UPDATE xau_gold_global_config SET parameter_value = 'true',
    change_reason = 'apr22 ablation: coupled with adx>=25, +1.48 PF'
    WHERE parameter_name = 'block_ranging';

UPDATE xau_gold_global_config SET parameter_value = '25.0',
    change_reason = 'apr22 ablation: regime gate partner for block_ranging'
    WHERE parameter_name = 'adx_trending_threshold';

UPDATE xau_gold_global_config SET parameter_value = '30',
    change_reason = 'apr22 ablation: 30 sufficient (180 killed PF by 0.25)'
    WHERE parameter_name = 'signal_cooldown_minutes';

-- Sample-size killers (DROP)
UPDATE xau_gold_global_config SET parameter_value = 'false',
    change_reason = 'apr22 ablation: -0.20 PF, gold trades outside London/NY'
    WHERE parameter_name = 'session_filter_enabled';

UPDATE xau_gold_global_config SET parameter_value = 'false',
    change_reason = 'apr22 ablation: -0.20 PF, sample-size killer'
    WHERE parameter_name = 'macd_filter_enabled';

UPDATE xau_gold_global_config SET parameter_value = 'false',
    change_reason = 'apr22 ablation: marginal -0.04 PF, not worth the 71 sig/mo'
    WHERE parameter_name = 'block_expansion';

UPDATE xau_gold_global_config SET parameter_value = '0.0',
    change_reason = 'apr22 ablation: fib zone 0.382-0.618 cost -0.33 PF'
    WHERE parameter_name = 'fib_pullback_min';

UPDATE xau_gold_global_config SET parameter_value = '1.5',
    change_reason = 'apr22 ablation: fib disabled (widened range)'
    WHERE parameter_name = 'fib_pullback_max';

-- Non-binding / scoring-only (NEUTRALIZE)
UPDATE xau_gold_global_config SET parameter_value = '0.0',
    change_reason = 'apr22 ablation: scoring ingredient not hard gate'
    WHERE parameter_name = 'rsi_neutral_min';

UPDATE xau_gold_global_config SET parameter_value = '100.0',
    change_reason = 'apr22 ablation: scoring ingredient not hard gate'
    WHERE parameter_name = 'rsi_neutral_max';

UPDATE xau_gold_global_config SET parameter_value = '0.0',
    change_reason = 'apr22 ablation: scoring ingredient not hard gate'
    WHERE parameter_name = 'bos_displacement_atr_mult';

UPDATE xau_gold_global_config SET parameter_value = '0.50',
    change_reason = 'apr22 ablation: base_confidence ceiling 0.72-0.75'
    WHERE parameter_name = 'min_confidence';

UPDATE xau_gold_global_config SET parameter_value = 'false',
    change_reason = 'apr22 ablation: dead code (hardcoded False in strategy)'
    WHERE parameter_name = 'dxy_confluence_enabled';

UPDATE xau_gold_global_config SET parameter_value = '100.0',
    change_reason = 'apr22 ablation: ATR expansion disabled via threshold'
    WHERE parameter_name = 'atr_expansion_pct';

-- ---------------------------------------------------------------------------
-- 2. PAIR OVERRIDES — flip to monitor-only on demo for 2-4w forward observation
-- ---------------------------------------------------------------------------
UPDATE xau_gold_pair_overrides
SET monitor_only = true,
    change_reason = 'apr22 lean-config launch: monitor-only forward-test period'
WHERE epic = 'CS.D.CFEGOLD.CEE.IP' AND config_set = 'demo';

-- Summary readout
SELECT 'xau_gold_global_config (demo)' AS scope, parameter_name, parameter_value
FROM xau_gold_global_config
WHERE config_set = 'demo' AND parameter_name IN (
    'require_ob_or_fvg','block_ranging','adx_trending_threshold',
    'signal_cooldown_minutes','session_filter_enabled','macd_filter_enabled',
    'fib_pullback_min','fib_pullback_max','block_expansion',
    'rsi_neutral_min','rsi_neutral_max','bos_displacement_atr_mult',
    'min_confidence','dxy_confluence_enabled','atr_expansion_pct'
) ORDER BY parameter_name;

SELECT 'xau_gold_pair_overrides' AS scope, config_set, epic, is_enabled, monitor_only
FROM xau_gold_pair_overrides ORDER BY config_set, epic;

COMMIT;
