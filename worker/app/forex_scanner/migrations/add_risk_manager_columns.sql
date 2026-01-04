-- Migration: Add RiskManager configuration columns to scanner_global_config
-- Date: 2026-01-04
-- Purpose: Enable database-only configuration for RiskManager (NO FALLBACK to config.py)
--
-- This migration adds all risk management settings required by RiskManager.
-- After this migration, RiskManager will FAIL if database is unavailable
-- rather than falling back to config.py defaults.

-- Add extended risk management columns
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS max_daily_loss_percent DOUBLE PRECISION DEFAULT 5.0;

COMMENT ON COLUMN scanner_global_config.max_daily_loss_percent IS
'Maximum daily loss as percentage of account balance. When reached, trading stops.';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS max_trades_per_pair INTEGER DEFAULT 3;

COMMENT ON COLUMN scanner_global_config.max_trades_per_pair IS
'Maximum number of trades allowed per currency pair per day.';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS min_account_balance DOUBLE PRECISION DEFAULT 1000.0;

COMMENT ON COLUMN scanner_global_config.min_account_balance IS
'Minimum required account balance to allow trading.';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS daily_profit_target_percent DOUBLE PRECISION DEFAULT 3.0;

COMMENT ON COLUMN scanner_global_config.daily_profit_target_percent IS
'Daily profit target as percentage of account balance.';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS stop_on_daily_target BOOLEAN DEFAULT false;

COMMENT ON COLUMN scanner_global_config.stop_on_daily_target IS
'If true, stop trading when daily profit target is reached.';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS testing_max_stop_percent DOUBLE PRECISION DEFAULT 20.0;

COMMENT ON COLUMN scanner_global_config.testing_max_stop_percent IS
'Maximum stop loss percentage allowed in testing mode (relaxed validation).';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS testing_min_confidence DOUBLE PRECISION DEFAULT 0.0;

COMMENT ON COLUMN scanner_global_config.testing_min_confidence IS
'Minimum confidence threshold in testing mode (0.0 = accept all).';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS emergency_stop_enabled BOOLEAN DEFAULT true;

COMMENT ON COLUMN scanner_global_config.emergency_stop_enabled IS
'Enable emergency stop functionality (halts trading on major loss events).';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS disable_account_risk_validation BOOLEAN DEFAULT false;

COMMENT ON COLUMN scanner_global_config.disable_account_risk_validation IS
'Bypass account risk validation (for testing only).';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS disable_position_sizing BOOLEAN DEFAULT false;

COMMENT ON COLUMN scanner_global_config.disable_position_sizing IS
'Use minimal position sizing instead of risk-based calculation (for testing).';

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS account_balance DOUBLE PRECISION DEFAULT 10000.0;

COMMENT ON COLUMN scanner_global_config.account_balance IS
'Account balance used for risk calculations. Should match actual account balance.';

-- Verify columns were added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'scanner_global_config'
AND column_name IN (
    'max_daily_loss_percent', 'max_trades_per_pair', 'min_account_balance',
    'daily_profit_target_percent', 'stop_on_daily_target', 'testing_max_stop_percent',
    'testing_min_confidence', 'emergency_stop_enabled', 'disable_account_risk_validation',
    'disable_position_sizing', 'account_balance'
)
ORDER BY column_name;
