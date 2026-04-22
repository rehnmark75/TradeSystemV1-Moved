-- Seed 5 pairs into RANGE_FADE pair overrides.
-- DEMO: is_enabled=t, monitor_only=f (execute orders), is_traded=t
-- LIVE: is_enabled=f (safety lock)

INSERT INTO eurusd_range_fade_pair_overrides (epic, pair_name, profile_name, config_set, is_enabled, is_traded, monitor_only, rsi_oversold, rsi_overbought, range_proximity_pips, max_current_range_pips, london_start_hour_utc, new_york_end_hour_utc, allow_neutral_htf, signal_cooldown_minutes, fixed_stop_loss_pips, fixed_take_profit_pips, parameter_overrides, notes) VALUES
('CS.D.GBPUSD.MINI.IP', 'GBPUSD', '5m', 'demo', true, true, false, 35, 65, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 180d BT: 73 signals WR 60.3% PF 2.25'),
('CS.D.GBPUSD.MINI.IP', 'GBPUSD', '5m', 'live', false, false, true, 35, 65, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 live disabled pending demo validation'),
('CS.D.USDCHF.MINI.IP', 'USDCHF', '5m', 'demo', true, true, false, 40, 60, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 180d BT: 130 signals WR 58.5% PF 2.17'),
('CS.D.USDCHF.MINI.IP', 'USDCHF', '5m', 'live', false, false, true, 40, 60, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 live disabled pending demo validation'),
('CS.D.USDCAD.MINI.IP', 'USDCAD', '5m', 'demo', true, true, false, 40, 60, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 180d BT: 121 signals WR 54.5% PF 1.83'),
('CS.D.USDCAD.MINI.IP', 'USDCAD', '5m', 'live', false, false, true, 40, 60, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 live disabled pending demo validation'),
('CS.D.NZDUSD.MINI.IP', 'NZDUSD', '5m', 'demo', true, true, false, 35, 65, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 180d BT: 54 signals WR 63.0% PF 2.48'),
('CS.D.NZDUSD.MINI.IP', 'NZDUSD', '5m', 'live', false, false, true, 35, 65, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 live disabled pending demo validation'),
('CS.D.AUDUSD.MINI.IP', 'AUDUSD', '5m', 'demo', true, true, false, 38, 62, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 180d BT: 88 signals WR 68.2% PF 3.17'),
('CS.D.AUDUSD.MINI.IP', 'AUDUSD', '5m', 'live', false, false, true, 38, 62, 999.00, 999.00, 6, 18, false, 0, 8.00, 12.00, '{"min_band_width_pips":0,"max_band_width_pips":9999}'::jsonb, '2026-04-22 live disabled pending demo validation')
ON CONFLICT (epic, profile_name, config_set) DO UPDATE SET
  is_enabled = EXCLUDED.is_enabled, is_traded = EXCLUDED.is_traded, monitor_only = EXCLUDED.monitor_only,
  rsi_oversold = EXCLUDED.rsi_oversold, rsi_overbought = EXCLUDED.rsi_overbought,
  range_proximity_pips = EXCLUDED.range_proximity_pips, max_current_range_pips = EXCLUDED.max_current_range_pips,
  london_start_hour_utc = EXCLUDED.london_start_hour_utc, new_york_end_hour_utc = EXCLUDED.new_york_end_hour_utc,
  allow_neutral_htf = EXCLUDED.allow_neutral_htf, signal_cooldown_minutes = EXCLUDED.signal_cooldown_minutes,
  fixed_stop_loss_pips = EXCLUDED.fixed_stop_loss_pips, fixed_take_profit_pips = EXCLUDED.fixed_take_profit_pips,
  parameter_overrides = EXCLUDED.parameter_overrides, notes = EXCLUDED.notes, updated_at = NOW();
