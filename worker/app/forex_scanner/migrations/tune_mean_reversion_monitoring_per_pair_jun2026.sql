-- ============================================================================
-- MEAN_REVERSION per-pair monitoring expansion (June 2026)
--
-- Goal:
--   Increase observable signal flow without increasing live execution risk.
--   All selected epics are set monitor_only=TRUE. Relaxation is per-pair, not
--   global, so later outcome analysis can compare each candidate independently.
--
-- Chosen cohort:
--   AUDUSD, NZDUSD, USDJPY: prior June 2026 production sanity checks.
--   EURJPY, USDCAD, USDCHF: strongest prior touch-mode / live-monitor evidence.
--   AUDJPY: probe only; positive 90d standalone but weak later production BT.
--
-- Deliberately not promoted:
--   GBPUSD and EURUSD had weak prior PF; leave existing state unless separately
--   tested in a lower-priority experiment.
-- ============================================================================

BEGIN;

-- AUDUSD: 15m rejection, ADX relaxation only. No session hard gate so we can
-- learn whether the time-of-day distribution is actually useful.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'rejection',
    session_start_hour = NULL,
    session_end_hour = NULL,
    low_vol_regime_filter_enabled = FALSE,
    adx_hard_ceiling_primary = 28.0,
    adx_hard_ceiling_htf = 32.0,
    bb_mult = 2.0,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_relaxed_adx',
            'experiment_notes', 'AUDUSD 15m rejection; ADX 22/25 -> 28/32; no hard session gate'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: AUDUSD relaxed ADX per-pair'
WHERE epic = 'CS.D.AUDUSD.MINI.IP'
  AND config_set = 'demo';

-- NZDUSD: keep validated touch mode, relax ADX, remove hard session for
-- monitoring so session can be evaluated as a feature rather than a blocker.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'touch',
    session_start_hour = NULL,
    session_end_hour = NULL,
    low_vol_regime_filter_enabled = FALSE,
    adx_hard_ceiling_primary = 28.0,
    adx_hard_ceiling_htf = 32.0,
    bb_mult = 2.0,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_relaxed_adx',
            'experiment_notes', 'NZDUSD 15m touch; ADX 22/25 -> 28/32; no hard session gate'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: NZDUSD relaxed ADX/session per-pair'
WHERE epic = 'CS.D.NZDUSD.MINI.IP'
  AND config_set = 'demo';

-- USDJPY: keep rejection mode, allow somewhat higher ADX because prior sanity
-- check favored this pair, but retain a broad London/NY session window.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'rejection',
    session_start_hour = 7,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = FALSE,
    adx_hard_ceiling_primary = 30.0,
    adx_hard_ceiling_htf = 35.0,
    bb_mult = 2.0,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_relaxed_adx',
            'experiment_notes', 'USDJPY 15m rejection; ADX 25/30 -> 30/35; session 7-22 UTC'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: USDJPY relaxed ADX/session per-pair'
WHERE epic = 'CS.D.USDJPY.MINI.IP'
  AND config_set = 'demo';

-- EURJPY: existing touch + low-vol thesis has live evidence. Keep low-vol
-- quality filter, widen session from 18-22 to 12-22, force monitor-only.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'touch',
    session_start_hour = 12,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = TRUE,
    fixed_stop_loss_pips = COALESCE(fixed_stop_loss_pips, 10.0),
    fixed_take_profit_pips = COALESCE(fixed_take_profit_pips, 7.0),
    primary_timeframe = '5m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 35,
            'rsi_overbought', 65,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_session_widen',
            'experiment_notes', 'EURJPY 5m touch + low-vol; session 18-22 -> 12-22 UTC'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: EURJPY widened session per-pair'
WHERE epic = 'CS.D.EURJPY.MINI.IP'
  AND config_set = 'demo';

-- USDCAD: prior touch-mode monitor evidence is strong. Keep low-vol quality
-- filter, widen session from 18-22 to 12-22.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'touch',
    session_start_hour = 12,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = TRUE,
    fixed_stop_loss_pips = COALESCE(fixed_stop_loss_pips, 10.0),
    fixed_take_profit_pips = COALESCE(fixed_take_profit_pips, 7.0),
    primary_timeframe = '5m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 35,
            'rsi_overbought', 65,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_session_widen',
            'experiment_notes', 'USDCAD 5m touch + low-vol; session 18-22 -> 12-22 UTC'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: USDCAD widened session per-pair'
WHERE epic = 'CS.D.USDCAD.MINI.IP'
  AND config_set = 'demo';

-- USDCHF: keep the researched late-session touch setup, but broaden monitoring
-- to catch earlier NY/overlap conditions. Low-vol filter stays on.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'touch',
    session_start_hour = 12,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = TRUE,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 7.0,
    primary_timeframe = '5m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 35,
            'rsi_overbought', 65,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_session_widen',
            'experiment_notes', 'USDCHF 5m touch + low-vol; session 18-22 -> 12-22 UTC'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: USDCHF widened session per-pair'
WHERE epic = 'CS.D.USDCHF.MINI.IP'
  AND config_set = 'demo';

-- AUDJPY: probe only. It had positive 90d standalone but weak later production
-- BT, so keep it monitor-only with moderate ADX and no low-vol substitution.
UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    entry_mode = 'rejection',
    session_start_hour = 7,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = FALSE,
    adx_hard_ceiling_primary = 28.0,
    adx_hard_ceiling_htf = 32.0,
    bb_mult = 2.0,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'disabled_reason', null,
            'monitoring_experiment', 'jun2026_per_pair_probe',
            'experiment_notes', 'AUDJPY probe; prior 90d standalone positive, later production BT weak; session 7-22 UTC'
        ),
    updated_by = 'codex',
    change_reason = 'MEAN_REVERSION monitor-only expansion: AUDJPY probe enabled'
WHERE epic = 'CS.D.AUDJPY.MINI.IP'
  AND config_set = 'demo';

COMMIT;

SELECT
    epic,
    config_set,
    is_enabled,
    monitor_only,
    entry_mode,
    session_start_hour,
    session_end_hour,
    low_vol_regime_filter_enabled,
    adx_hard_ceiling_primary,
    adx_hard_ceiling_htf,
    bb_mult,
    fixed_stop_loss_pips,
    fixed_take_profit_pips,
    primary_timeframe,
    parameter_overrides ->> 'monitoring_experiment' AS monitoring_experiment
FROM mean_reversion_pair_overrides
WHERE config_set = 'demo'
ORDER BY epic;
