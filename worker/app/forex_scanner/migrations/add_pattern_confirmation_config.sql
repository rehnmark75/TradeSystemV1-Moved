-- ============================================================================
-- PATTERN CONFIRMATION & TRIGGER TYPE TRACKING MIGRATION
-- Adds alternative trigger patterns (price action, RSI divergence, MACD alignment)
-- ============================================================================
-- Purpose: Increase signal frequency via price action patterns while maintaining
--          quality through MONITORING mode validation
-- Patterns: Pin bar, engulfing, inside bar, RSI divergence, MACD alignment
-- ============================================================================

-- ============================================================================
-- PHASE 1: PRICE ACTION PATTERN CONFIRMATION
-- ============================================================================

-- Master toggle for pattern confirmation system
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_confirmation_enabled BOOLEAN DEFAULT FALSE;

-- Mode: MONITORING (logs only) or ACTIVE (boosts confidence/allows marginal entries)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_confirmation_mode VARCHAR(20) DEFAULT 'MONITORING';

-- Minimum pattern strength to consider (0.0-1.0)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_min_strength DECIMAL(4,3) DEFAULT 0.70;

-- Individual pattern toggles
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_pin_bar_enabled BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_engulfing_enabled BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_inside_bar_enabled BOOLEAN DEFAULT TRUE;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_hammer_shooter_enabled BOOLEAN DEFAULT TRUE;

-- Confidence boost when pattern is detected (added to base confidence)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS pattern_confidence_boost DECIMAL(4,3) DEFAULT 0.05;

-- ============================================================================
-- PHASE 2: RSI DIVERGENCE DETECTION
-- ============================================================================

-- Master toggle for RSI divergence detection
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS rsi_divergence_enabled BOOLEAN DEFAULT FALSE;

-- Mode: MONITORING or ACTIVE
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS rsi_divergence_mode VARCHAR(20) DEFAULT 'MONITORING';

-- Lookback bars for divergence detection
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS rsi_divergence_lookback INTEGER DEFAULT 20;

-- Minimum divergence strength (0.0-1.0)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS rsi_divergence_min_strength DECIMAL(4,3) DEFAULT 0.30;

-- Confidence boost when divergence detected
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS rsi_divergence_confidence_boost DECIMAL(4,3) DEFAULT 0.08;

-- ============================================================================
-- PHASE 3: MACD ALIGNMENT ENHANCEMENT
-- ============================================================================

-- Master toggle for MACD alignment check (added to TIER 1)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS macd_alignment_enabled BOOLEAN DEFAULT FALSE;

-- If TRUE, signals are rejected when MACD not aligned (stricter)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS macd_alignment_required BOOLEAN DEFAULT FALSE;

-- Confidence boost when MACD is aligned
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS macd_alignment_confidence_boost DECIMAL(4,3) DEFAULT 0.05;

-- ============================================================================
-- PHASE 4: PATTERN TRACKING COLUMNS IN alert_history
-- For easier querying of pattern performance
-- ============================================================================

-- Pattern type detected (if any)
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(30);

-- Pattern strength (0.0-1.0)
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS pattern_strength DECIMAL(5,4);

-- RSI divergence detected
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS rsi_divergence_detected BOOLEAN DEFAULT FALSE;

-- MACD alignment status
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS macd_aligned BOOLEAN;

-- Create index for pattern analysis
CREATE INDEX IF NOT EXISTS idx_alert_history_pattern_type
    ON alert_history(pattern_type);

CREATE INDEX IF NOT EXISTS idx_alert_history_trigger_pattern
    ON alert_history(trigger_type, pattern_type);

-- ============================================================================
-- ANALYSIS VIEW: Pattern Performance
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_performance_by_trigger AS
SELECT
    trigger_type,
    pattern_type,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN order_status = 'filled' THEN 1 END) as executed,
    AVG(confidence_score) as avg_confidence,
    AVG(pattern_strength) as avg_pattern_strength,
    COUNT(CASE WHEN rsi_divergence_detected THEN 1 END) as with_divergence,
    COUNT(CASE WHEN macd_aligned THEN 1 END) as with_macd_aligned
FROM alert_history
WHERE alert_timestamp > NOW() - INTERVAL '30 days'
  AND strategy = 'smc_simple'
GROUP BY trigger_type, pattern_type
ORDER BY total_signals DESC;

-- ============================================================================
-- ANALYSIS VIEW: Baseline vs Pattern-Enhanced Comparison
-- ============================================================================

CREATE OR REPLACE VIEW v_pattern_enhanced_comparison AS
SELECT
    CASE
        WHEN trigger_type LIKE '%+%' THEN 'Pattern Enhanced'
        WHEN pattern_type IS NOT NULL THEN 'Pattern Confirmed'
        ELSE 'Baseline Swing'
    END as category,
    trigger_type,
    COUNT(*) as signals,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN order_status = 'filled' THEN 1 END) as filled_count
FROM alert_history
WHERE alert_timestamp > NOW() - INTERVAL '14 days'
  AND strategy = 'smc_simple'
GROUP BY
    CASE
        WHEN trigger_type LIKE '%+%' THEN 'Pattern Enhanced'
        WHEN pattern_type IS NOT NULL THEN 'Pattern Confirmed'
        ELSE 'Baseline Swing'
    END,
    trigger_type
ORDER BY category, signals DESC;

-- ============================================================================
-- PARAMETER METADATA: Add UI descriptions for new parameters
-- ============================================================================

INSERT INTO smc_simple_parameter_metadata (parameter_name, display_name, category, data_type, default_value, description)
VALUES
    -- Pattern confirmation
    ('pattern_confirmation_enabled', 'Pattern Confirmation', 'Alternative Triggers', 'boolean', 'false',
     'Enable price action pattern detection (pin bar, engulfing, etc.) for entry confirmation'),
    ('pattern_confirmation_mode', 'Pattern Mode', 'Alternative Triggers', 'string', 'MONITORING',
     'MONITORING = log patterns only, ACTIVE = boost confidence when pattern detected'),
    ('pattern_min_strength', 'Min Pattern Strength', 'Alternative Triggers', 'decimal', '0.70',
     'Minimum pattern strength (0.0-1.0) required for consideration. 0.70 = 70% strength'),
    ('pattern_pin_bar_enabled', 'Pin Bar Pattern', 'Alternative Triggers', 'boolean', 'true',
     'Detect pin bar / rejection candle patterns'),
    ('pattern_engulfing_enabled', 'Engulfing Pattern', 'Alternative Triggers', 'boolean', 'true',
     'Detect bullish/bearish engulfing patterns'),
    ('pattern_inside_bar_enabled', 'Inside Bar Pattern', 'Alternative Triggers', 'boolean', 'true',
     'Detect inside bar consolidation patterns'),
    ('pattern_confidence_boost', 'Pattern Confidence Boost', 'Alternative Triggers', 'decimal', '0.05',
     'Confidence boost (0.0-1.0) added when pattern detected. 0.05 = +5%'),

    -- RSI divergence
    ('rsi_divergence_enabled', 'RSI Divergence', 'Alternative Triggers', 'boolean', 'false',
     'Enable RSI divergence detection for momentum confirmation'),
    ('rsi_divergence_mode', 'Divergence Mode', 'Alternative Triggers', 'string', 'MONITORING',
     'MONITORING = log divergences only, ACTIVE = boost confidence when detected'),
    ('rsi_divergence_lookback', 'Divergence Lookback', 'Alternative Triggers', 'integer', '20',
     'Number of bars to look back for divergence detection'),
    ('rsi_divergence_confidence_boost', 'Divergence Confidence Boost', 'Alternative Triggers', 'decimal', '0.08',
     'Confidence boost when RSI divergence detected. 0.08 = +8%'),

    -- MACD alignment
    ('macd_alignment_enabled', 'MACD Alignment', 'Alternative Triggers', 'boolean', 'false',
     'Enable MACD histogram alignment check in TIER 1'),
    ('macd_alignment_required', 'MACD Required', 'Alternative Triggers', 'boolean', 'false',
     'If TRUE, reject signals when MACD not aligned (stricter mode)'),
    ('macd_alignment_confidence_boost', 'MACD Confidence Boost', 'Alternative Triggers', 'decimal', '0.05',
     'Confidence boost when MACD is aligned. 0.05 = +5%')
ON CONFLICT (parameter_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    category = EXCLUDED.category,
    description = EXCLUDED.description;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN smc_simple_global_config.pattern_confirmation_enabled IS
'Enable price action pattern detection for entry confirmation.
Patterns: pin bar, engulfing, hammer/shooting star, inside bar.
Start in MONITORING mode to validate before going ACTIVE.';

COMMENT ON COLUMN smc_simple_global_config.rsi_divergence_enabled IS
'Enable RSI divergence detection for momentum confirmation.
Bullish divergence: price lower low, RSI higher low.
Bearish divergence: price higher high, RSI lower high.';

COMMENT ON COLUMN smc_simple_global_config.macd_alignment_enabled IS
'Enable MACD histogram alignment check in TIER 1.
BULL: histogram > 0 and rising. BEAR: histogram < 0 and falling.
When macd_alignment_required=TRUE, non-aligned signals are rejected.';

COMMENT ON COLUMN alert_history.trigger_type IS
'Signal trigger mechanism. Values:
- SWING_PULLBACK: Standard Fib pullback entry (23.6%-70%)
- SWING_MOMENTUM: Momentum continuation (-20% to 0%)
- SWING_OPTIMAL: Optimal Fib zone (38.2%-61.8%)
- Pattern-enhanced: SWING_PULLBACK+PIN, SWING_OPTIMAL+ENG, etc.';

COMMENT ON COLUMN alert_history.pattern_type IS
'Price action pattern detected at entry.
Values: bullish_pin_bar, bearish_pin_bar, bullish_engulfing, bearish_engulfing,
hammer, shooting_star, inside_bar, NULL (no pattern)';

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
    'pattern_confirmation_config',
    'N/A',
    'Added pattern confirmation and alternative trigger system',
    'migration',
    'Add price action patterns (pin bar, engulfing), RSI divergence, MACD alignment for increased signal frequency'
FROM smc_simple_global_config
WHERE is_active = TRUE;

-- ============================================================================
-- VERIFICATION QUERIES (run after migration)
-- ============================================================================

-- Verify pattern config columns added
-- SELECT column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'smc_simple_global_config'
--   AND column_name LIKE 'pattern_%' OR column_name LIKE 'rsi_divergence%' OR column_name LIKE 'macd_alignment%';

-- Verify alert_history columns added
-- SELECT column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'alert_history'
--   AND column_name IN ('pattern_type', 'pattern_strength', 'rsi_divergence_detected', 'macd_aligned');

-- Verify views created
-- SELECT * FROM information_schema.views WHERE table_name LIKE 'v_signal_performance%' OR table_name LIKE 'v_pattern%';
