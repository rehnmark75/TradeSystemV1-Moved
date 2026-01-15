-- ============================================================================
-- SCALP MICRO-REGIME VALIDATION CONFIGURATION MIGRATION
-- Adds immediate price action filters for scalp trades
-- ============================================================================
-- Purpose: Enable micro-regime validation to filter scalp signals based on
--          immediate price action (last 3-5 candles) rather than slower indicators
-- Expected outcome: Improved win rate by avoiding entries in choppy/congested markets
-- ============================================================================

-- ============================================================================
-- GLOBAL CONFIG: Add micro-regime columns
-- ============================================================================

-- Master toggle for micro-regime validation
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_micro_regime_enabled BOOLEAN DEFAULT FALSE;

-- ============================================================================
-- PER-FILTER TOGGLES
-- ============================================================================

-- Consecutive Candles Filter: Require N candles in trade direction
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_consecutive_candles_enabled BOOLEAN DEFAULT TRUE;

-- Anti-Chop Filter: Reject alternating green/red patterns
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_anti_chop_enabled BOOLEAN DEFAULT TRUE;

-- Body Dominance Filter: Require body > wick (conviction)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_body_dominance_enabled BOOLEAN DEFAULT TRUE;

-- Micro-Range Filter: Reject tight congestion zones
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_micro_range_enabled BOOLEAN DEFAULT TRUE;

-- Momentum Candle Filter: Require strong thrust candle
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_momentum_candle_enabled BOOLEAN DEFAULT FALSE;

-- ============================================================================
-- THRESHOLDS
-- ============================================================================

-- Consecutive candles minimum (e.g., 2 = require 2+ aligned candles)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_consecutive_candles_min INTEGER DEFAULT 2;

-- Anti-chop lookback (how many candles to check for alternation)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_anti_chop_lookback INTEGER DEFAULT 4;

-- Anti-chop max alternations allowed (e.g., 2 = allow up to 2 direction changes)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_anti_chop_max_alternations INTEGER DEFAULT 2;

-- Body dominance lookback (candles to average)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_body_dominance_lookback INTEGER DEFAULT 3;

-- Body dominance ratio (1.0 = body must equal wick, 1.5 = body must be 1.5x wick)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_body_dominance_ratio DECIMAL(4,2) DEFAULT 1.0;

-- Micro-range lookback (candles to measure range)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_micro_range_lookback INTEGER DEFAULT 5;

-- Micro-range minimum pips (below this = congestion, reject)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_micro_range_min_pips DECIMAL(5,2) DEFAULT 3.0;

-- Momentum candle multiplier (1.5 = last candle body must be 1.5x average)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_momentum_candle_multiplier DECIMAL(4,2) DEFAULT 1.5;

-- ============================================================================
-- ADD MICRO-REGIME RESULTS TO QUALIFICATION LOG
-- ============================================================================

-- Consecutive candles filter results
ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS consecutive_candles_passed BOOLEAN;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS consecutive_candles_count INTEGER;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS consecutive_candles_reason TEXT;

-- Anti-chop filter results
ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS anti_chop_passed BOOLEAN;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS anti_chop_alternations INTEGER;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS anti_chop_reason TEXT;

-- Body dominance filter results
ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS body_dominance_passed BOOLEAN;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS body_dominance_ratio DECIMAL(5,2);

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS body_dominance_reason TEXT;

-- Micro-range filter results
ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS micro_range_passed BOOLEAN;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS micro_range_pips DECIMAL(6,2);

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS micro_range_reason TEXT;

-- Momentum candle filter results
ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS momentum_candle_passed BOOLEAN;

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS momentum_candle_multiplier DECIMAL(5,2);

ALTER TABLE scalp_qualification_log
ADD COLUMN IF NOT EXISTS momentum_candle_reason TEXT;

-- ============================================================================
-- PARAMETER METADATA: Add UI descriptions for new parameters
-- ============================================================================

INSERT INTO smc_simple_parameter_metadata (parameter_name, display_name, category, data_type, default_value, description)
VALUES
    ('scalp_micro_regime_enabled', 'Micro-Regime Enabled', 'Scalp Micro-Regime', 'boolean', 'false',
     'Enable micro-regime validation (analyzes last 3-5 candles for immediate price action quality)'),
    ('scalp_consecutive_candles_enabled', 'Consecutive Candles', 'Scalp Micro-Regime', 'boolean', 'true',
     'Require last N candles to align with trade direction'),
    ('scalp_anti_chop_enabled', 'Anti-Chop Filter', 'Scalp Micro-Regime', 'boolean', 'true',
     'Reject signals when market shows alternating green/red pattern (choppy)'),
    ('scalp_body_dominance_enabled', 'Body Dominance', 'Scalp Micro-Regime', 'boolean', 'true',
     'Require candle bodies to dominate over wicks (shows conviction)'),
    ('scalp_micro_range_enabled', 'Micro-Range Filter', 'Scalp Micro-Regime', 'boolean', 'true',
     'Reject signals when recent candles are in tight range (congestion)'),
    ('scalp_momentum_candle_enabled', 'Momentum Candle', 'Scalp Micro-Regime', 'boolean', 'false',
     'Require last candle to show momentum thrust (body > 1.5x average)'),
    ('scalp_consecutive_candles_min', 'Min Consecutive', 'Scalp Micro-Regime', 'integer', '2',
     'Minimum consecutive candles required in trade direction'),
    ('scalp_anti_chop_lookback', 'Anti-Chop Lookback', 'Scalp Micro-Regime', 'integer', '4',
     'Number of candles to check for alternating pattern'),
    ('scalp_anti_chop_max_alternations', 'Max Alternations', 'Scalp Micro-Regime', 'integer', '2',
     'Maximum allowed direction changes before rejecting as choppy'),
    ('scalp_body_dominance_ratio', 'Body/Wick Ratio', 'Scalp Micro-Regime', 'decimal', '1.0',
     'Required ratio of average body size to average wick size'),
    ('scalp_micro_range_min_pips', 'Min Range Pips', 'Scalp Micro-Regime', 'decimal', '3.0',
     'Minimum range in pips for last N candles (below = congestion)'),
    ('scalp_momentum_candle_multiplier', 'Momentum Multiplier', 'Scalp Micro-Regime', 'decimal', '1.5',
     'Required body size as multiple of average (e.g., 1.5 = 150% of average)')
ON CONFLICT (parameter_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    category = EXCLUDED.category,
    description = EXCLUDED.description;

-- ============================================================================
-- COMPOSITE INDEX FOR MICRO-REGIME ANALYSIS
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_scalp_qual_micro_regime
    ON scalp_qualification_log(
        consecutive_candles_passed,
        anti_chop_passed,
        body_dominance_passed,
        micro_range_passed,
        trade_outcome
    );

-- ============================================================================
-- ANALYSIS VIEW: Micro-Regime Filter Effectiveness
-- ============================================================================

CREATE OR REPLACE VIEW v_scalp_micro_regime_effectiveness AS
SELECT
    'CONSECUTIVE_CANDLES' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN consecutive_candles_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN consecutive_candles_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN consecutive_candles_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN consecutive_candles_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT consecutive_candles_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT consecutive_candles_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE consecutive_candles_passed IS NOT NULL AND trade_outcome IN ('WIN', 'LOSS')

UNION ALL

SELECT
    'ANTI_CHOP' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN anti_chop_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN anti_chop_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN anti_chop_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN anti_chop_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT anti_chop_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT anti_chop_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE anti_chop_passed IS NOT NULL AND trade_outcome IN ('WIN', 'LOSS')

UNION ALL

SELECT
    'BODY_DOMINANCE' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN body_dominance_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN body_dominance_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN body_dominance_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN body_dominance_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT body_dominance_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT body_dominance_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE body_dominance_passed IS NOT NULL AND trade_outcome IN ('WIN', 'LOSS')

UNION ALL

SELECT
    'MICRO_RANGE' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN micro_range_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN micro_range_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN micro_range_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN micro_range_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT micro_range_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT micro_range_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE micro_range_passed IS NOT NULL AND trade_outcome IN ('WIN', 'LOSS')

UNION ALL

SELECT
    'MOMENTUM_CANDLE' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN momentum_candle_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN momentum_candle_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN momentum_candle_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN momentum_candle_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT momentum_candle_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT momentum_candle_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE momentum_candle_passed IS NOT NULL AND trade_outcome IN ('WIN', 'LOSS');

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN smc_simple_global_config.scalp_micro_regime_enabled IS
'Master toggle for micro-regime validation. When enabled, analyzes immediate price action
(last 3-5 candles) to filter signals in choppy or congested market conditions.';

COMMENT ON COLUMN smc_simple_global_config.scalp_consecutive_candles_enabled IS
'Require last N candles to align with trade direction. For BUY: last 2+ candles green.
For SELL: last 2+ candles red. Prevents entries where price just reversed.';

COMMENT ON COLUMN smc_simple_global_config.scalp_anti_chop_enabled IS
'Reject signals when market shows alternating green/red candles (choppy market).
Choppy markets make scalp entries unreliable as price bounces unpredictably.';

COMMENT ON COLUMN smc_simple_global_config.scalp_body_dominance_enabled IS
'Require candle bodies to dominate over wicks. High body/wick ratio = conviction.
Low ratio = indecision and potential reversals. Default ratio 1.0 = body >= wick.';

COMMENT ON COLUMN smc_simple_global_config.scalp_micro_range_enabled IS
'Reject signals when recent candles are in tight range (congestion/consolidation).
Scalp entries in congestion often get stopped out as price bounces within range.';

COMMENT ON COLUMN smc_simple_global_config.scalp_momentum_candle_enabled IS
'Require last candle to show momentum thrust (body > 1.5x average body size).
Confirms strong conviction in trade direction. More strict filter, disabled by default.';

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
    'scalp_micro_regime_config',
    'N/A',
    'Added micro-regime validation filters',
    'migration',
    'Add immediate price action filters (consecutive candles, anti-chop, body dominance, micro-range, momentum candle)'
FROM smc_simple_global_config
WHERE is_active = TRUE;
