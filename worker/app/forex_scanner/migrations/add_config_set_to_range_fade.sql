-- Add demo/live environment split to RANGE_FADE tables
-- Each strategy must support independent enable-state per environment (demo vs live).
-- Existing rows become the 'demo' set; a 'live' copy is seeded with identical values
-- so operators can diverge them from the UI afterwards.

BEGIN;

-- ========== Global config ==========
ALTER TABLE eurusd_range_fade_global_config
  ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'demo';

ALTER TABLE eurusd_range_fade_global_config
  DROP CONSTRAINT IF EXISTS eurusd_range_fade_global_config_profile_name_is_active_key;

ALTER TABLE eurusd_range_fade_global_config
  ADD CONSTRAINT eurusd_range_fade_global_config_profile_config_active_key
  UNIQUE (profile_name, config_set, is_active);

INSERT INTO eurusd_range_fade_global_config (
  profile_name, is_active, strategy_name, version, monitor_only,
  primary_timeframe, confirmation_timeframe,
  bb_period, bb_mult, rsi_period, rsi_oversold, rsi_overbought,
  range_lookback_bars, range_proximity_pips, min_band_width_pips, max_band_width_pips,
  htf_ema_period, htf_slope_bars, allow_neutral_htf, max_current_range_pips,
  min_confidence, max_confidence, fixed_stop_loss_pips, fixed_take_profit_pips,
  signal_cooldown_minutes, london_start_hour_utc, new_york_end_hour_utc,
  notes, config_set
)
SELECT
  profile_name, is_active, strategy_name, version, monitor_only,
  primary_timeframe, confirmation_timeframe,
  bb_period, bb_mult, rsi_period, rsi_oversold, rsi_overbought,
  range_lookback_bars, range_proximity_pips, min_band_width_pips, max_band_width_pips,
  htf_ema_period, htf_slope_bars, allow_neutral_htf, max_current_range_pips,
  min_confidence, max_confidence, fixed_stop_loss_pips, fixed_take_profit_pips,
  signal_cooldown_minutes, london_start_hour_utc, new_york_end_hour_utc,
  COALESCE(notes, '') || E'\n2026-04-21: seeded live config_set from demo',
  'live'
FROM eurusd_range_fade_global_config
WHERE config_set = 'demo' AND is_active = TRUE
ON CONFLICT (profile_name, config_set, is_active) DO NOTHING;

-- ========== Pair overrides ==========
ALTER TABLE eurusd_range_fade_pair_overrides
  ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'demo';

ALTER TABLE eurusd_range_fade_pair_overrides
  DROP CONSTRAINT IF EXISTS eurusd_range_fade_pair_overrides_epic_profile_name_key;

ALTER TABLE eurusd_range_fade_pair_overrides
  ADD CONSTRAINT eurusd_range_fade_pair_overrides_epic_profile_config_key
  UNIQUE (epic, profile_name, config_set);

INSERT INTO eurusd_range_fade_pair_overrides (
  epic, pair_name, profile_name, is_enabled, is_traded, monitor_only,
  signal_cooldown_minutes, rsi_oversold, rsi_overbought,
  range_lookback_bars, range_proximity_pips, max_current_range_pips,
  fixed_stop_loss_pips, fixed_take_profit_pips,
  london_start_hour_utc, new_york_end_hour_utc, allow_neutral_htf,
  parameter_overrides, notes, disabled_reason, config_set
)
SELECT
  epic, pair_name, profile_name, is_enabled, is_traded, monitor_only,
  signal_cooldown_minutes, rsi_oversold, rsi_overbought,
  range_lookback_bars, range_proximity_pips, max_current_range_pips,
  fixed_stop_loss_pips, fixed_take_profit_pips,
  london_start_hour_utc, new_york_end_hour_utc, allow_neutral_htf,
  parameter_overrides,
  COALESCE(notes, '') || E'\n2026-04-21: seeded live config_set from demo',
  disabled_reason, 'live'
FROM eurusd_range_fade_pair_overrides
WHERE config_set = 'demo'
ON CONFLICT (epic, profile_name, config_set) DO NOTHING;

COMMIT;
