-- Migration: Add HTF Bias Score Configuration
-- Date: 2026-01-29
-- Description: Replaces binary HTF alignment filter with continuous bias score system
--
-- This migration adds:
-- 1. Global HTF bias configuration parameters
-- 2. Per-pair HTF bias mode (active/monitor/disabled)
-- 3. New columns for storing bias scores in alert_history
--
-- Backward Compatibility:
-- - Old parameters (scalp_require_htf_alignment, scalp_reversal_*) remain but are deprecated
-- - New system uses htf_bias_* parameters

-- ============================================================================
-- Part 1: Add HTF Bias columns to smc_simple_global_config
-- ============================================================================

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS htf_bias_enabled BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS htf_bias_min_threshold DECIMAL(4,3) DEFAULT 0.400;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS htf_bias_confidence_multiplier_enabled BOOLEAN DEFAULT TRUE;

-- Comment on new columns
COMMENT ON COLUMN smc_simple_global_config.htf_bias_enabled IS
'Master toggle for HTF bias score system (replaces scalp_require_htf_alignment)';

COMMENT ON COLUMN smc_simple_global_config.htf_bias_min_threshold IS
'Minimum HTF bias score required (0.0-1.0). Signals below this are filtered in active mode. Default 0.4';

COMMENT ON COLUMN smc_simple_global_config.htf_bias_confidence_multiplier_enabled IS
'When true, HTF bias score adjusts signal confidence (0.7-1.3x multiplier)';

-- ============================================================================
-- Part 2: Add HTF Bias mode to per-pair overrides
-- ============================================================================

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS htf_bias_mode VARCHAR(20) DEFAULT 'active';

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS htf_bias_min_threshold DECIMAL(4,3) DEFAULT NULL;

COMMENT ON COLUMN smc_simple_pair_overrides.htf_bias_mode IS
'Per-pair HTF bias mode: active (filter signals), monitor (log only), disabled';

COMMENT ON COLUMN smc_simple_pair_overrides.htf_bias_min_threshold IS
'Per-pair threshold override. NULL = use global default';

-- ============================================================================
-- Part 3: Add HTF Bias score columns to alert_history
-- ============================================================================

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS htf_bias_score DECIMAL(4,3) DEFAULT NULL;

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS htf_bias_mode VARCHAR(20) DEFAULT NULL;

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS htf_bias_details JSONB DEFAULT NULL;

COMMENT ON COLUMN alert_history.htf_bias_score IS
'HTF bias alignment score (0.0-1.0) at signal generation time';

COMMENT ON COLUMN alert_history.htf_bias_mode IS
'HTF bias mode active at signal time: active, monitor, disabled';

COMMENT ON COLUMN alert_history.htf_bias_details IS
'Full HTF bias calculation details (candle, EMA, MACD components)';

-- Index for analysis
CREATE INDEX IF NOT EXISTS idx_alert_history_htf_bias_score
ON alert_history(htf_bias_score) WHERE htf_bias_score IS NOT NULL;

-- ============================================================================
-- Part 4: Update default values for active configuration
-- ============================================================================

-- Set global defaults
UPDATE smc_simple_global_config
SET
    htf_bias_enabled = TRUE,
    htf_bias_min_threshold = 0.400,
    htf_bias_confidence_multiplier_enabled = TRUE
WHERE is_active = TRUE;

-- ============================================================================
-- Part 5: Set per-pair HTF bias modes based on current scalp_require_htf_alignment
-- Pairs with HTF currently disabled -> monitoring mode
-- Pairs with HTF currently enabled -> active mode
-- ============================================================================

-- EURUSD, USDCAD, USDJPY have scalp_require_htf_alignment = false -> set to 'monitor'
UPDATE smc_simple_pair_overrides
SET htf_bias_mode = 'monitor'
WHERE epic IN ('CS.D.EURUSD.CEEM.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.USDJPY.MINI.IP');

-- Other pairs inherit global = true -> set to 'active'
UPDATE smc_simple_pair_overrides
SET htf_bias_mode = 'active'
WHERE epic NOT IN ('CS.D.EURUSD.CEEM.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.USDJPY.MINI.IP');

-- ============================================================================
-- Verification queries
-- ============================================================================

-- Show global config
-- SELECT htf_bias_enabled, htf_bias_min_threshold, htf_bias_confidence_multiplier_enabled
-- FROM smc_simple_global_config WHERE is_active = TRUE;

-- Show per-pair modes
-- SELECT epic, htf_bias_mode, htf_bias_min_threshold FROM smc_simple_pair_overrides ORDER BY epic;
