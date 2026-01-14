-- ============================================================================
-- SCALP MODE CONFIGURATION MIGRATION
-- Adds high-frequency scalping parameters to SMC Simple strategy
-- ============================================================================
-- Purpose: Enable a "scalp mode" toggle that transforms the strategy for
--          high-frequency trading with 5 pip TP targets
-- Expected outcome: 10-20+ signals/day vs current 0.07 signals/day
-- ============================================================================

-- ============================================================================
-- GLOBAL CONFIG: Add scalp mode columns
-- ============================================================================

-- Master toggle for scalp mode
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_mode_enabled BOOLEAN DEFAULT FALSE;

-- Scalp SL/TP settings (1:1 R:R by default)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_tp_pips DECIMAL(4,1) DEFAULT 5.0;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_sl_pips DECIMAL(4,1) DEFAULT 5.0;

-- Spread filter (critical for scalping profitability)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_max_spread_pips DECIMAL(4,2) DEFAULT 1.0;

-- Scalp timeframes (faster than swing mode)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_htf_timeframe VARCHAR(10) DEFAULT '15m';

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_trigger_timeframe VARCHAR(10) DEFAULT '5m';

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_entry_timeframe VARCHAR(10) DEFAULT '1m';

-- Scalp EMA settings (faster EMA for quicker signals)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_ema_period INTEGER DEFAULT 20;

-- Scalp confidence threshold (lower to allow more entries)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_min_confidence DECIMAL(4,3) DEFAULT 0.30;

-- Scalp filter disables (relax filters for more signals)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_disable_ema_slope_validation BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_disable_swing_proximity BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_disable_volume_filter BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_disable_macd_filter BOOLEAN DEFAULT TRUE;

-- Scalp entry logic settings
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_use_momentum_only BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_momentum_min_depth DECIMAL(5,3) DEFAULT -0.30;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_fib_pullback_min DECIMAL(5,3) DEFAULT 0.0;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_fib_pullback_max DECIMAL(5,3) DEFAULT 1.0;

-- Scalp cooldown (much shorter than swing mode)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_cooldown_minutes INTEGER DEFAULT 15;

-- Scalp spread filter toggle (24/7 trading with spread gate)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_require_tight_spread BOOLEAN DEFAULT TRUE;

-- Scalp swing detection settings (faster detection)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_swing_lookback_bars INTEGER DEFAULT 5;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_range_position_threshold DECIMAL(4,2) DEFAULT 0.80;

-- Scalp Claude AI integration (enable AI validation for scalp trades)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_enable_claude_ai BOOLEAN DEFAULT TRUE;

-- ============================================================================
-- PER-PAIR OVERRIDES: Add scalp-specific columns
-- ============================================================================

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_enabled BOOLEAN;

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_tp_pips DECIMAL(4,1);

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_sl_pips DECIMAL(4,1);

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_max_spread_pips DECIMAL(4,2);

-- ============================================================================
-- PARAMETER METADATA: Add UI descriptions for new parameters
-- ============================================================================

INSERT INTO smc_simple_parameter_metadata (parameter_name, display_name, category, data_type, default_value, description)
VALUES
    ('scalp_mode_enabled', 'Scalp Mode', 'Scalp', 'boolean', 'false', 'Enable high-frequency scalping mode with relaxed filters and 5 pip TP'),
    ('scalp_tp_pips', 'Scalp Take Profit', 'Scalp', 'decimal', '5.0', 'Take profit target for scalp trades (pips)'),
    ('scalp_sl_pips', 'Scalp Stop Loss', 'Scalp', 'decimal', '5.0', 'Stop loss for scalp trades (pips) - 1:1 R:R'),
    ('scalp_max_spread_pips', 'Max Spread Filter', 'Scalp', 'decimal', '1.0', 'Maximum spread allowed for scalp entries (pips)'),
    ('scalp_htf_timeframe', 'Scalp HTF', 'Scalp', 'string', '15m', 'Higher timeframe for scalp mode bias'),
    ('scalp_trigger_timeframe', 'Scalp Trigger TF', 'Scalp', 'string', '5m', 'Trigger timeframe for scalp mode'),
    ('scalp_entry_timeframe', 'Scalp Entry TF', 'Scalp', 'string', '1m', 'Entry timeframe for scalp mode'),
    ('scalp_ema_period', 'Scalp EMA Period', 'Scalp', 'integer', '20', 'Faster EMA period for scalp mode'),
    ('scalp_min_confidence', 'Scalp Min Confidence', 'Scalp', 'decimal', '0.30', 'Lower confidence threshold for more scalp entries'),
    ('scalp_cooldown_minutes', 'Scalp Cooldown', 'Scalp', 'integer', '15', 'Cooldown between scalp trades (minutes)'),
    ('scalp_require_tight_spread', 'Require Tight Spread', 'Scalp', 'boolean', 'true', 'Only trade when spread is below max threshold'),
    ('scalp_enable_claude_ai', 'Enable Claude AI', 'Scalp', 'boolean', 'true', 'Enable Claude AI validation for scalp trades')
ON CONFLICT (parameter_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    category = EXCLUDED.category,
    description = EXCLUDED.description;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN smc_simple_global_config.scalp_mode_enabled IS
'Master toggle for high-frequency scalping mode. When enabled, uses faster timeframes (1H/5m/1m),
5 pip TP target, relaxed filters, and 15-minute cooldown. Requires spread < 1 pip for entries.';

COMMENT ON COLUMN smc_simple_global_config.scalp_max_spread_pips IS
'Critical for scalping profitability. With 5 pip TP, a 1 pip spread eats 20% of profit.
Trades are blocked when spread exceeds this threshold.';

COMMENT ON COLUMN smc_simple_global_config.scalp_require_tight_spread IS
'When TRUE, enables 24/7 trading with spread-gated entries.
Trades only execute when current spread is below scalp_max_spread_pips.';

-- ============================================================================
-- AUDIT ENTRY
-- ============================================================================

INSERT INTO smc_simple_config_audit (
    config_id,
    changed_parameter,
    old_value,
    new_value,
    changed_by,
    change_reason
)
SELECT
    id,
    'scalp_mode_config',
    'N/A',
    'Added scalp mode configuration columns',
    'migration',
    'Add high-frequency scalping mode with 5 pip TP, spread filter, faster timeframes'
FROM smc_simple_global_config
WHERE is_active = TRUE;
