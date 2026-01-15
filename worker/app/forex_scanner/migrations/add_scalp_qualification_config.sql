-- ============================================================================
-- SCALP SIGNAL QUALIFICATION CONFIGURATION MIGRATION
-- Adds momentum confirmation filters for scalp trades
-- ============================================================================
-- Purpose: Enable signal qualification to improve scalp trade win rate
--          by filtering out low-quality entries based on momentum confirmation
-- Expected outcome: Win rate improvement from ~35% to 50%+
-- ============================================================================

-- ============================================================================
-- GLOBAL CONFIG: Add qualification columns
-- ============================================================================

-- Master toggle for qualification system
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_qualification_enabled BOOLEAN DEFAULT FALSE;

-- Mode: MONITORING (logs only) or ACTIVE (blocks signals)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_qualification_mode VARCHAR(20) DEFAULT 'MONITORING';

-- Minimum score required to pass (0.0-1.0)
-- 0.50 = require at least 50% of filters to pass
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_min_qualification_score DECIMAL(4,3) DEFAULT 0.50;

-- ============================================================================
-- PER-FILTER TOGGLES
-- ============================================================================

-- RSI Momentum Filter: Validates RSI zone and direction
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_rsi_filter_enabled BOOLEAN DEFAULT TRUE;

-- Two-Pole Oscillator Filter: Minimal-lag momentum confirmation
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_two_pole_filter_enabled BOOLEAN DEFAULT TRUE;

-- MACD Direction Filter: Histogram momentum confirmation
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_macd_filter_enabled BOOLEAN DEFAULT TRUE;

-- ============================================================================
-- RSI THRESHOLDS
-- ============================================================================

-- RSI thresholds for BULL signals (buy zone)
-- BUY: RSI must be between bull_min and bull_max, AND rising
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_rsi_bull_min INTEGER DEFAULT 40;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_rsi_bull_max INTEGER DEFAULT 75;

-- RSI thresholds for BEAR signals (sell zone)
-- SELL: RSI must be between bear_min and bear_max, AND falling
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_rsi_bear_min INTEGER DEFAULT 25;

ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_rsi_bear_max INTEGER DEFAULT 60;

-- ============================================================================
-- TWO-POLE OSCILLATOR THRESHOLDS
-- ============================================================================

-- Two-Pole threshold for BULL confirmation (oversold zone)
-- Values below this are considered oversold (good for recovery buys)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_two_pole_bull_threshold DECIMAL(4,3) DEFAULT -0.30;

-- Two-Pole threshold for BEAR confirmation (overbought zone)
-- Values above this are considered overbought (good for reversal sells)
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS scalp_two_pole_bear_threshold DECIMAL(4,3) DEFAULT 0.30;

-- ============================================================================
-- QUALIFICATION LOG TABLE
-- Tracks qualification results for analysis and optimization
-- ============================================================================

CREATE TABLE IF NOT EXISTS scalp_qualification_log (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Signal identification
    epic VARCHAR(50) NOT NULL,
    pair VARCHAR(20),
    direction VARCHAR(4) NOT NULL,  -- BULL or BEAR
    signal_timestamp TIMESTAMP WITH TIME ZONE,

    -- Qualification results
    qualification_score DECIMAL(4,3),
    qualification_mode VARCHAR(20),  -- MONITORING or ACTIVE
    signal_blocked BOOLEAN DEFAULT FALSE,

    -- RSI filter results
    rsi_passed BOOLEAN,
    rsi_value DECIMAL(6,2),
    rsi_prev DECIMAL(6,2),
    rsi_reason TEXT,

    -- Two-Pole filter results
    two_pole_passed BOOLEAN,
    two_pole_value DECIMAL(8,5),
    two_pole_is_green BOOLEAN,
    two_pole_is_purple BOOLEAN,
    two_pole_reason TEXT,

    -- MACD filter results
    macd_passed BOOLEAN,
    macd_histogram DECIMAL(12,8),
    macd_histogram_prev DECIMAL(12,8),
    macd_reason TEXT,

    -- Trade outcome (updated later by trade monitoring)
    trade_outcome VARCHAR(10),  -- WIN, LOSS, EXPIRED, PENDING
    pnl_pips DECIMAL(8,2),
    trade_id VARCHAR(50),

    -- Entry details
    entry_price DECIMAL(12,6),
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    entry_type VARCHAR(20),  -- MOMENTUM or PULLBACK

    -- Market context
    spread_at_signal DECIMAL(6,3),
    atr_at_signal DECIMAL(12,6),
    ema_distance_pips DECIMAL(8,2)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_scalp_qual_created
    ON scalp_qualification_log(created_at);

CREATE INDEX IF NOT EXISTS idx_scalp_qual_epic
    ON scalp_qualification_log(epic);

CREATE INDEX IF NOT EXISTS idx_scalp_qual_outcome
    ON scalp_qualification_log(trade_outcome);

CREATE INDEX IF NOT EXISTS idx_scalp_qual_score
    ON scalp_qualification_log(qualification_score);

CREATE INDEX IF NOT EXISTS idx_scalp_qual_blocked
    ON scalp_qualification_log(signal_blocked);

-- Composite index for filter analysis
CREATE INDEX IF NOT EXISTS idx_scalp_qual_filters
    ON scalp_qualification_log(rsi_passed, two_pole_passed, macd_passed, trade_outcome);

-- ============================================================================
-- PARAMETER METADATA: Add UI descriptions for new parameters
-- ============================================================================

INSERT INTO smc_simple_parameter_metadata (parameter_name, display_name, category, data_type, default_value, description)
VALUES
    ('scalp_qualification_enabled', 'Qualification Enabled', 'Scalp Qualification', 'boolean', 'false',
     'Enable signal qualification system to filter low-quality scalp entries'),
    ('scalp_qualification_mode', 'Qualification Mode', 'Scalp Qualification', 'string', 'MONITORING',
     'MONITORING = log results only, ACTIVE = block signals below threshold'),
    ('scalp_min_qualification_score', 'Min Qualification Score', 'Scalp Qualification', 'decimal', '0.50',
     'Minimum score (0.0-1.0) required to pass in ACTIVE mode. 0.50 = 50% of filters must pass'),
    ('scalp_rsi_filter_enabled', 'RSI Filter', 'Scalp Qualification', 'boolean', 'true',
     'Enable RSI momentum filter (validates RSI zone and direction)'),
    ('scalp_two_pole_filter_enabled', 'Two-Pole Filter', 'Scalp Qualification', 'boolean', 'true',
     'Enable Two-Pole oscillator filter (minimal-lag momentum confirmation)'),
    ('scalp_macd_filter_enabled', 'MACD Filter', 'Scalp Qualification', 'boolean', 'true',
     'Enable MACD histogram direction filter'),
    ('scalp_rsi_bull_min', 'RSI Bull Min', 'Scalp Qualification', 'integer', '40',
     'Minimum RSI for BULL signals (below = weak momentum)'),
    ('scalp_rsi_bull_max', 'RSI Bull Max', 'Scalp Qualification', 'integer', '75',
     'Maximum RSI for BULL signals (above = overbought exhaustion)'),
    ('scalp_rsi_bear_min', 'RSI Bear Min', 'Scalp Qualification', 'integer', '25',
     'Minimum RSI for BEAR signals (below = oversold exhaustion)'),
    ('scalp_rsi_bear_max', 'RSI Bear Max', 'Scalp Qualification', 'integer', '60',
     'Maximum RSI for BEAR signals (above = weak momentum)')
ON CONFLICT (parameter_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    category = EXCLUDED.category,
    description = EXCLUDED.description;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN smc_simple_global_config.scalp_qualification_enabled IS
'Master toggle for signal qualification system. When enabled, runs momentum filters
on all scalp signals to improve win rate by filtering low-quality entries.';

COMMENT ON COLUMN smc_simple_global_config.scalp_qualification_mode IS
'Operating mode: MONITORING = logs filter results but passes all signals (for analysis).
ACTIVE = blocks signals that don''t meet minimum qualification score.';

COMMENT ON COLUMN smc_simple_global_config.scalp_min_qualification_score IS
'Minimum qualification score (0.0-1.0) required in ACTIVE mode.
Score = proportion of filters passed. 0.50 means 50% of enabled filters must pass.';

COMMENT ON TABLE scalp_qualification_log IS
'Tracks qualification results for all scalp signals. Used to analyze filter effectiveness
and optimize thresholds. Trade outcomes are updated by trade monitoring system.';

-- ============================================================================
-- ANALYSIS VIEWS
-- ============================================================================

-- View: Filter effectiveness analysis
CREATE OR REPLACE VIEW v_scalp_qualification_filter_effectiveness AS
SELECT
    -- RSI Filter
    'RSI' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN rsi_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN rsi_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN rsi_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN rsi_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT rsi_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT rsi_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE trade_outcome IN ('WIN', 'LOSS')

UNION ALL

SELECT
    -- Two-Pole Filter
    'TWO_POLE' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN two_pole_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN two_pole_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN two_pole_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN two_pole_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT two_pole_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT two_pole_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE trade_outcome IN ('WIN', 'LOSS')

UNION ALL

SELECT
    -- MACD Filter
    'MACD' as filter_name,
    COUNT(*) as total_signals,
    SUM(CASE WHEN macd_passed THEN 1 ELSE 0 END) as pass_count,
    ROUND(100.0 * SUM(CASE WHEN macd_passed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN macd_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN macd_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_passed,
    ROUND(100.0 * SUM(CASE WHEN NOT macd_passed AND trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN NOT macd_passed AND trade_outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0), 1) as win_rate_when_failed
FROM scalp_qualification_log
WHERE trade_outcome IN ('WIN', 'LOSS');

-- View: Qualification score vs outcome analysis
CREATE OR REPLACE VIEW v_scalp_qualification_score_analysis AS
SELECT
    CASE
        WHEN qualification_score >= 1.0 THEN '100% (All Pass)'
        WHEN qualification_score >= 0.66 THEN '66-99% (Most Pass)'
        WHEN qualification_score >= 0.33 THEN '33-65% (Some Pass)'
        ELSE '0-32% (Few Pass)'
    END as score_bucket,
    COUNT(*) as total_signals,
    SUM(CASE WHEN trade_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN trade_outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
    ROUND(100.0 * SUM(CASE WHEN trade_outcome = 'WIN' THEN 1 ELSE 0 END) /
          NULLIF(COUNT(*), 0), 1) as win_rate_pct,
    ROUND(AVG(pnl_pips), 2) as avg_pnl_pips,
    ROUND(SUM(pnl_pips), 2) as total_pnl_pips
FROM scalp_qualification_log
WHERE trade_outcome IN ('WIN', 'LOSS')
GROUP BY
    CASE
        WHEN qualification_score >= 1.0 THEN '100% (All Pass)'
        WHEN qualification_score >= 0.66 THEN '66-99% (Most Pass)'
        WHEN qualification_score >= 0.33 THEN '33-65% (Some Pass)'
        ELSE '0-32% (Few Pass)'
    END
ORDER BY score_bucket DESC;

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
    'scalp_qualification_config',
    'N/A',
    'Added scalp signal qualification system',
    'migration',
    'Add momentum confirmation filters (RSI, Two-Pole, MACD) for scalp trade qualification'
FROM smc_simple_global_config
WHERE is_active = TRUE;

-- ============================================================================
-- VERIFICATION QUERIES (run after migration)
-- ============================================================================

-- Verify columns added
-- SELECT column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'smc_simple_global_config'
--   AND column_name LIKE 'scalp_qualification%' OR column_name LIKE 'scalp_rsi%' OR column_name LIKE 'scalp_two_pole%';

-- Verify table created
-- SELECT * FROM information_schema.tables WHERE table_name = 'scalp_qualification_log';

-- Verify views created
-- SELECT * FROM information_schema.views WHERE table_name LIKE 'v_scalp_qualification%';
