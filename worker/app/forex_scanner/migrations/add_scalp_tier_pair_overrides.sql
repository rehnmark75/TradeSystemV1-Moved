-- ============================================================================
-- SCALP MODE PER-PAIR TIER SETTINGS MIGRATION
-- Adds per-pair scalp tier configuration for optimized settings per currency pair
-- ============================================================================
-- Purpose: Allow different scalp mode parameters for each pair based on
--          pair-specific characteristics (volatility, spread patterns, etc.)
-- Example: EURUSD might use EMA 30, swing lookback 8
--          GBPJPY might use EMA 20, swing lookback 15 (more volatile)
-- ============================================================================

-- ============================================================================
-- PER-PAIR SCALP TIER SETTINGS
-- ============================================================================

-- Scalp EMA period override (TIER 1: Trend filter)
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_ema_period INTEGER;

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_ema_period IS
'Per-pair EMA period for scalp mode TIER 1 trend filter. Default: 20.
Lower values (10-15) = more responsive but more noise.
Higher values (30-50) = smoother trend but fewer signals.';

-- Scalp swing lookback bars override (TIER 2: Structure detection)
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_swing_lookback_bars INTEGER;

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_swing_lookback_bars IS
'Per-pair swing lookback bars for scalp mode TIER 2 structure detection. Default: 12.
Lower values (5-8) = detect smaller swings, more signals.
Higher values (15-20) = detect larger swings, fewer but stronger signals.';

-- Scalp limit order offset override (entry timing)
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_limit_offset_pips DECIMAL(4,2);

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_limit_offset_pips IS
'Per-pair limit order offset for scalp mode momentum confirmation. Default: 1 pip.
Lower values = faster fills but more slippage risk.
Higher values = better entries but more missed trades.';

-- Scalp HTF timeframe override (TIER 1: Higher timeframe)
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_htf_timeframe VARCHAR(10);

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_htf_timeframe IS
'Per-pair higher timeframe for scalp mode TIER 1 bias. Default: 15m.
Options: 5m, 15m, 30m, 1h. Lower = more signals, higher = stronger trend filter.';

-- Scalp trigger timeframe override (TIER 2: Structure timeframe)
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_trigger_timeframe VARCHAR(10);

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_trigger_timeframe IS
'Per-pair trigger timeframe for scalp mode TIER 2 swing detection. Default: 5m.
Options: 1m, 5m, 15m. Should be lower than HTF.';

-- Scalp entry timeframe override (TIER 3: Entry timeframe)
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_entry_timeframe VARCHAR(10);

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_entry_timeframe IS
'Per-pair entry timeframe for scalp mode TIER 3 pullback detection. Default: 1m.
Options: 1m, 5m. Lower = more precise entries.';

-- Scalp minimum confidence override
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_min_confidence DECIMAL(4,3);

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_min_confidence IS
'Per-pair minimum confidence threshold for scalp mode. Default: 0.30.
Lower values = more signals, higher values = stricter filtering.';

-- Scalp cooldown minutes override
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_cooldown_minutes INTEGER;

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_cooldown_minutes IS
'Per-pair cooldown between scalp trades in minutes. Default: 15.
Lower = more frequent trading, higher = more rest between trades.';

-- Scalp swing break tolerance override
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS scalp_swing_break_tolerance_pips DECIMAL(4,2);

COMMENT ON COLUMN smc_simple_pair_overrides.scalp_swing_break_tolerance_pips IS
'Per-pair tolerance for near-swing-breaks in scalp mode. Default: 0.5 pips.
Allows entries when price is very close to but not quite breaking swing level.';

-- ============================================================================
-- PARAMETER METADATA: Add UI descriptions for new per-pair parameters
-- ============================================================================

INSERT INTO smc_simple_parameter_metadata (parameter_name, display_name, category, data_type, default_value, description)
VALUES
    ('pair_scalp_ema_period', 'Pair Scalp EMA', 'Scalp Per-Pair', 'integer', 'NULL', 'Per-pair EMA period for scalp mode (NULL = use global)'),
    ('pair_scalp_swing_lookback_bars', 'Pair Scalp Swing Lookback', 'Scalp Per-Pair', 'integer', 'NULL', 'Per-pair swing lookback bars for scalp mode (NULL = use global)'),
    ('pair_scalp_limit_offset_pips', 'Pair Scalp Offset', 'Scalp Per-Pair', 'decimal', 'NULL', 'Per-pair limit order offset for scalp mode (NULL = use global)'),
    ('pair_scalp_htf_timeframe', 'Pair Scalp HTF', 'Scalp Per-Pair', 'string', 'NULL', 'Per-pair HTF timeframe for scalp mode (NULL = use global)'),
    ('pair_scalp_min_confidence', 'Pair Scalp Confidence', 'Scalp Per-Pair', 'decimal', 'NULL', 'Per-pair minimum confidence for scalp mode (NULL = use global)'),
    ('pair_scalp_cooldown_minutes', 'Pair Scalp Cooldown', 'Scalp Per-Pair', 'integer', 'NULL', 'Per-pair cooldown for scalp mode (NULL = use global)')
ON CONFLICT (parameter_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    category = EXCLUDED.category,
    description = EXCLUDED.description;

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
    'scalp_tier_pair_overrides',
    'N/A',
    'Added per-pair scalp tier columns: ema_period, swing_lookback, offset, htf, trigger, entry timeframes, confidence, cooldown, tolerance',
    'migration',
    'Enable per-pair scalp mode tier optimization based on backtesting results'
FROM smc_simple_global_config
WHERE is_active = TRUE;

-- ============================================================================
-- EXAMPLE: Set optimized scalp settings for EURUSD
-- ============================================================================
-- Uncomment and run to set EURUSD-specific scalp settings:
--
-- UPDATE smc_simple_pair_overrides
-- SET
--     scalp_ema_period = 30,
--     scalp_swing_lookback_bars = 8,
--     scalp_limit_offset_pips = 1.0,
--     scalp_htf_timeframe = '15m'
-- WHERE epic = 'CS.D.EURUSD.CEEM.IP';
--
-- ============================================================================
