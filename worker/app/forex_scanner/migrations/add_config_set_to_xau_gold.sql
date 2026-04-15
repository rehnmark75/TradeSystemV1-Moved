-- Add demo/live scoping to XAU_GOLD configuration tables.

BEGIN;

ALTER TABLE xau_gold_global_config
  ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live',
  ADD COLUMN IF NOT EXISTS updated_by VARCHAR(100),
  ADD COLUMN IF NOT EXISTS change_reason TEXT;

ALTER TABLE xau_gold_global_config
  DROP CONSTRAINT IF EXISTS xau_gold_global_config_parameter_name_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_xau_gold_global_config_set_param
  ON xau_gold_global_config(config_set, parameter_name);

ALTER TABLE xau_gold_pair_overrides
  ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live',
  ADD COLUMN IF NOT EXISTS updated_by VARCHAR(100),
  ADD COLUMN IF NOT EXISTS change_reason TEXT;

ALTER TABLE xau_gold_pair_overrides
  DROP CONSTRAINT IF EXISTS xau_gold_pair_overrides_epic_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_xau_gold_pair_scope
  ON xau_gold_pair_overrides(config_set, epic);

ALTER TABLE xau_gold_config_audit
  ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live';

UPDATE xau_gold_global_config
SET config_set = 'live'
WHERE config_set IS NULL OR config_set = '';

UPDATE xau_gold_pair_overrides
SET config_set = 'live'
WHERE config_set IS NULL OR config_set = '';

UPDATE xau_gold_config_audit
SET config_set = 'live'
WHERE config_set IS NULL OR config_set = '';

INSERT INTO xau_gold_global_config (
  parameter_name, parameter_value, value_type, category, display_order, description,
  is_active, is_editable, created_at, updated_at, config_set, updated_by, change_reason
)
SELECT
  parameter_name, parameter_value, value_type, category, display_order, description,
  is_active, is_editable, created_at, updated_at, 'demo', 'migration_add_config_set_to_xau_gold',
  'Cloned from live config'
FROM xau_gold_global_config
WHERE config_set = 'live'
ON CONFLICT (config_set, parameter_name) DO NOTHING;

INSERT INTO xau_gold_pair_overrides (
  epic, pair_name, pip_size, fixed_stop_loss_pips, fixed_take_profit_pips,
  min_confidence, max_confidence, sl_atr_multiplier, rr_ratio, signal_cooldown_minutes,
  parameter_overrides, is_enabled, is_traded, monitor_only, notes,
  created_at, updated_at, config_set, updated_by, change_reason
)
SELECT
  epic, pair_name, pip_size, fixed_stop_loss_pips, fixed_take_profit_pips,
  min_confidence, max_confidence, sl_atr_multiplier, rr_ratio, signal_cooldown_minutes,
  parameter_overrides, is_enabled, is_traded, monitor_only, notes,
  created_at, updated_at, 'demo', 'migration_add_config_set_to_xau_gold',
  'Cloned from live config'
FROM xau_gold_pair_overrides
WHERE config_set = 'live'
ON CONFLICT (config_set, epic) DO NOTHING;

COMMIT;
