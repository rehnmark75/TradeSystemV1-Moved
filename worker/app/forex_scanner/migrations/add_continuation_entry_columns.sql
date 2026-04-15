-- v5.0.0: Continuation Entry columns for SMC Simple strategy
-- Adds opt-in per-pair continuation entry support (no pullback required in strong trends).
-- All columns default to safe/off values; per-pair tuning via parameter_overrides JSONB.
ALTER TABLE smc_simple_global_config
  ADD COLUMN IF NOT EXISTS continuation_entry_enabled BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS continuation_entry_min_adx NUMERIC DEFAULT 25.0,
  ADD COLUMN IF NOT EXISTS continuation_entry_min_efficiency NUMERIC DEFAULT 0.40,
  ADD COLUMN IF NOT EXISTS continuation_entry_max_extension_atr NUMERIC DEFAULT 1.5,
  ADD COLUMN IF NOT EXISTS continuation_entry_min_depth NUMERIC DEFAULT -1.0,
  ADD COLUMN IF NOT EXISTS continuation_entry_max_bars_since_break INT DEFAULT 8,
  ADD COLUMN IF NOT EXISTS continuation_entry_sl_atr_multiple NUMERIC DEFAULT 1.0,
  ADD COLUMN IF NOT EXISTS continuation_entry_sl_min_pips NUMERIC DEFAULT 8.0,
  ADD COLUMN IF NOT EXISTS continuation_entry_sl_max_pips NUMERIC DEFAULT 35.0;

-- Per-pair overrides use the existing parameter_overrides JSONB column — no schema change needed.
-- To enable for a specific pair:
--   UPDATE smc_simple_pair_overrides
--     SET parameter_overrides = parameter_overrides || '{"continuation_entry_enabled": true}'::jsonb
--     WHERE epic = 'CS.D.EURUSD.CEEM.IP';
