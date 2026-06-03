-- ============================================================================
-- MEAN_REVERSION pair candidate tuning (June 2026)
--
-- Basis:
--   - 365d direct-data grid over 5m canonical candles resampled to 15m.
--   - Production sanity check for AUDUSD over 180d:
--       22 signals, 59.1% win rate, roughly PF 1.31 from reported avg win/loss.
--   - Production sanity check for NZDUSD over 180d:
--       20 signals, 60.0% win rate, roughly PF 1.20 from reported avg win/loss.
--   - Production sanity check for USDJPY over 180d:
--       13 signals, 69.2% win rate, roughly PF 1.82 from reported avg win/loss.
--
-- Scope:
--   Pair-level overrides only. Keep monitor_only=TRUE pending live/forward review.
--
-- Run:
--   docker exec -i postgres psql -U postgres -d strategy_config \
--     -f /tmp/tune_mean_reversion_pair_candidates_jun2026.sql
-- ============================================================================

UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    adx_hard_ceiling_primary = 22.0,
    adx_hard_ceiling_htf = 25.0,
    bb_mult = 2.0,
    entry_mode = 'rejection',
    session_start_hour = NULL,
    session_end_hour = NULL,
    low_vol_regime_filter_enabled = FALSE,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'disabled_reason', NULL,
            'candidate_basis', 'Jun 2026 365d direct grid + 180d production sanity: AUDUSD rejection 15m ADX22/25 RSI30/70 SLTP10/15'
        ),
    updated_by = 'codex',
    change_reason = 'AUDUSD MEAN_REVERSION monitor-only candidate: pair-level rejection config validated by Jun 2026 analysis'
WHERE epic = 'CS.D.AUDUSD.MINI.IP'
  AND config_set = 'demo';

UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    adx_hard_ceiling_primary = 22.0,
    adx_hard_ceiling_htf = 25.0,
    bb_mult = 2.0,
    entry_mode = 'touch',
    session_start_hour = 12,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = FALSE,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'candidate_basis', 'Jun 2026 365d direct grid + 180d production sanity: NZDUSD touch 15m ADX22/25 RSI30/70 session12-22 SLTP10/15; prod 20 signals WR60 roughPF1.20'
        ),
    updated_by = 'codex',
    change_reason = 'NZDUSD MEAN_REVERSION monitor-only candidate: session-gated touch config validated by Jun 2026 analysis'
WHERE epic = 'CS.D.NZDUSD.MINI.IP'
  AND config_set = 'demo';

UPDATE mean_reversion_pair_overrides
SET
    is_enabled = TRUE,
    monitor_only = TRUE,
    fixed_stop_loss_pips = 10.0,
    fixed_take_profit_pips = 15.0,
    adx_hard_ceiling_primary = 25.0,
    adx_hard_ceiling_htf = 30.0,
    bb_mult = 2.0,
    entry_mode = 'rejection',
    session_start_hour = 12,
    session_end_hour = 22,
    low_vol_regime_filter_enabled = FALSE,
    primary_timeframe = '15m',
    parameter_overrides =
        COALESCE(parameter_overrides, '{}'::jsonb)
        || jsonb_build_object(
            'rsi_oversold', 30,
            'rsi_overbought', 70,
            'monitor_only', true,
            'candidate_basis', 'Jun 2026 365d direct grid + 180d production sanity: USDJPY rejection 15m ADX25/30 RSI30/70 session12-22 SLTP10/15; prod 13 signals WR69.2 roughPF1.82'
        ),
    updated_by = 'codex',
    change_reason = 'USDJPY MEAN_REVERSION monitor-only candidate: session-gated rejection config validated by Jun 2026 analysis'
WHERE epic = 'CS.D.USDJPY.MINI.IP'
  AND config_set = 'demo';

SELECT
    epic,
    config_set,
    is_enabled,
    monitor_only,
    fixed_stop_loss_pips,
    fixed_take_profit_pips,
    adx_hard_ceiling_primary,
    adx_hard_ceiling_htf,
    bb_mult,
    entry_mode,
    session_start_hour,
    session_end_hour,
    low_vol_regime_filter_enabled,
    primary_timeframe,
    parameter_overrides
FROM mean_reversion_pair_overrides
WHERE epic IN (
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP'
)
ORDER BY epic, config_set;
