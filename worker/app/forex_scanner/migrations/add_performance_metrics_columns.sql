-- ============================================================================
-- PERFORMANCE METRICS COLUMNS MIGRATION
-- ============================================================================
-- Purpose: Add enhanced performance metrics columns to alert_history and
--          smc_simple_rejections tables for improved trade analysis.
--
-- Metrics Added:
-- - Kaufman Efficiency Ratio (ER) - trend quality measurement
-- - Market Regime Classification - trending/ranging/breakout/high_vol
-- - Bollinger Band Width Percentile - volatility context
-- - Entry Quality Score - distance from optimal Fib zone
-- - Multi-Timeframe Confluence - alignment across timeframes
-- - Volume Profile Metrics - volume quality at key points
--
-- Created: 2025-01-01
-- Version: 1.0.0
-- ============================================================================

-- ============================================================================
-- ALERT HISTORY TABLE - Performance Metrics
-- ============================================================================

-- Kaufman Efficiency Ratio
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS efficiency_ratio DECIMAL(6,4);

COMMENT ON COLUMN alert_history.efficiency_ratio IS
    'Kaufman Efficiency Ratio (0-1): 1.0=perfect trend, 0.0=choppy. >0.6 trending, <0.3 ranging';

-- Market Regime Classification
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS market_regime_detected VARCHAR(20);

COMMENT ON COLUMN alert_history.market_regime_detected IS
    'Detected market regime: trending, ranging, breakout, high_volatility, unknown';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS regime_confidence DECIMAL(5,4);

COMMENT ON COLUMN alert_history.regime_confidence IS
    'Confidence score (0-1) for market regime classification';

-- Volatility Context
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS bb_width_percentile DECIMAL(5,2);

COMMENT ON COLUMN alert_history.bb_width_percentile IS
    'Current Bollinger Band width as percentile of 50-period history (0-100)';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS atr_percentile DECIMAL(5,2);

COMMENT ON COLUMN alert_history.atr_percentile IS
    'Current ATR as percentile of 20-period history (0-100)';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS volatility_state VARCHAR(15);

COMMENT ON COLUMN alert_history.volatility_state IS
    'Volatility classification: low, normal, high, extreme';

-- Entry Quality
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS entry_quality_score DECIMAL(5,4);

COMMENT ON COLUMN alert_history.entry_quality_score IS
    'Entry quality score (0-1) based on Fib zone accuracy and candle momentum';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS distance_from_optimal_fib DECIMAL(6,4);

COMMENT ON COLUMN alert_history.distance_from_optimal_fib IS
    'Distance from optimal Fib zone (38.2-50%). 0 = in zone, positive = outside';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS entry_candle_momentum DECIMAL(5,4);

COMMENT ON COLUMN alert_history.entry_candle_momentum IS
    'Entry candle body as percentage of range (0-1). Higher = more momentum';

-- Multi-Timeframe Confluence
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS mtf_confluence_score DECIMAL(5,4);

COMMENT ON COLUMN alert_history.mtf_confluence_score IS
    'Multi-timeframe alignment score (0-1). 1.0 = all timeframes aligned';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS htf_candle_position VARCHAR(10);

COMMENT ON COLUMN alert_history.htf_candle_position IS
    'Position within 4H candle at entry: start, middle, end';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS all_timeframes_aligned BOOLEAN;

COMMENT ON COLUMN alert_history.all_timeframes_aligned IS
    'True if 5m, 15m, and 4H all aligned with signal direction';

-- Volume Profile
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS volume_at_swing_break DECIMAL(8,4);

COMMENT ON COLUMN alert_history.volume_at_swing_break IS
    'Volume ratio at Tier 2 swing break (vs 20-period SMA)';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS volume_trend VARCHAR(15);

COMMENT ON COLUMN alert_history.volume_trend IS
    'Volume trend over last 10 bars: increasing, decreasing, stable';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS volume_quality_score DECIMAL(5,4);

COMMENT ON COLUMN alert_history.volume_quality_score IS
    'Overall volume quality score (0-1)';

-- ADX Components
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS adx_value DECIMAL(6,2);

COMMENT ON COLUMN alert_history.adx_value IS
    'ADX value at signal time. >25=trending, <20=ranging';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS adx_plus_di DECIMAL(6,2);

COMMENT ON COLUMN alert_history.adx_plus_di IS
    '+DI component of ADX';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS adx_minus_di DECIMAL(6,2);

COMMENT ON COLUMN alert_history.adx_minus_di IS
    '-DI component of ADX';

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS adx_trend_strength VARCHAR(15);

COMMENT ON COLUMN alert_history.adx_trend_strength IS
    'ADX trend strength classification: weak, moderate, strong, extreme';

-- Full metrics JSON (for any additional data)
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS performance_metrics JSON;

COMMENT ON COLUMN alert_history.performance_metrics IS
    'Full performance metrics object as JSON for future extensibility';


-- ============================================================================
-- SMC SIMPLE REJECTIONS TABLE - Performance Metrics
-- ============================================================================

-- Kaufman Efficiency Ratio
ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS efficiency_ratio DECIMAL(6,4);

COMMENT ON COLUMN smc_simple_rejections.efficiency_ratio IS
    'Kaufman Efficiency Ratio (0-1) at rejection time';

-- Market Regime
ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS market_regime_detected VARCHAR(20);

COMMENT ON COLUMN smc_simple_rejections.market_regime_detected IS
    'Detected market regime at rejection: trending, ranging, breakout, high_volatility';

-- Volatility Context
ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS bb_width_percentile DECIMAL(5,2);

COMMENT ON COLUMN smc_simple_rejections.bb_width_percentile IS
    'Bollinger Band width percentile (0-100) at rejection';

ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS volatility_state VARCHAR(15);

COMMENT ON COLUMN smc_simple_rejections.volatility_state IS
    'Volatility state at rejection: low, normal, high, extreme';

-- ADX for regime analysis
ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS adx_value DECIMAL(6,2);

COMMENT ON COLUMN smc_simple_rejections.adx_value IS
    'ADX value at rejection time';

ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS adx_trend_strength VARCHAR(15);

COMMENT ON COLUMN smc_simple_rejections.adx_trend_strength IS
    'ADX trend strength at rejection';

-- Full metrics JSON
ALTER TABLE smc_simple_rejections
ADD COLUMN IF NOT EXISTS performance_metrics JSON;

COMMENT ON COLUMN smc_simple_rejections.performance_metrics IS
    'Full performance metrics object as JSON';


-- ============================================================================
-- INDEXES FOR PERFORMANCE QUERIES
-- ============================================================================

-- Alert history indexes
CREATE INDEX IF NOT EXISTS idx_alert_history_efficiency_ratio
    ON alert_history(efficiency_ratio);

CREATE INDEX IF NOT EXISTS idx_alert_history_market_regime
    ON alert_history(market_regime_detected);

CREATE INDEX IF NOT EXISTS idx_alert_history_volatility_state
    ON alert_history(volatility_state);

CREATE INDEX IF NOT EXISTS idx_alert_history_entry_quality
    ON alert_history(entry_quality_score);

CREATE INDEX IF NOT EXISTS idx_alert_history_mtf_confluence
    ON alert_history(mtf_confluence_score);

-- Composite indexes for common analysis queries
CREATE INDEX IF NOT EXISTS idx_alert_history_regime_er
    ON alert_history(market_regime_detected, efficiency_ratio);

CREATE INDEX IF NOT EXISTS idx_alert_history_vol_quality
    ON alert_history(volume_quality_score, volatility_state);

-- SMC rejections indexes
CREATE INDEX IF NOT EXISTS idx_smc_rejections_efficiency_ratio
    ON smc_simple_rejections(efficiency_ratio);

CREATE INDEX IF NOT EXISTS idx_smc_rejections_market_regime
    ON smc_simple_rejections(market_regime_detected);

CREATE INDEX IF NOT EXISTS idx_smc_rejections_volatility_state
    ON smc_simple_rejections(volatility_state);


-- ============================================================================
-- ANALYSIS VIEW: Signal Performance by Market Regime
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_performance_by_regime AS
SELECT
    market_regime_detected as regime,
    COUNT(*) as total_signals,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    ROUND(AVG(efficiency_ratio)::numeric, 4) as avg_efficiency_ratio,
    ROUND(AVG(entry_quality_score)::numeric, 3) as avg_entry_quality,
    ROUND(AVG(mtf_confluence_score)::numeric, 3) as avg_mtf_confluence,
    ROUND(AVG(volume_quality_score)::numeric, 3) as avg_volume_quality,
    COUNT(CASE WHEN claude_approved = true THEN 1 END) as claude_approved,
    ROUND(
        COUNT(CASE WHEN claude_approved = true THEN 1 END)::numeric /
        NULLIF(COUNT(*), 0) * 100, 1
    ) as approval_rate_pct
FROM alert_history
WHERE alert_timestamp >= NOW() - INTERVAL '30 days'
  AND market_regime_detected IS NOT NULL
GROUP BY market_regime_detected
ORDER BY total_signals DESC;

COMMENT ON VIEW v_signal_performance_by_regime IS
    'Signal performance metrics aggregated by detected market regime (last 30 days)';


-- ============================================================================
-- ANALYSIS VIEW: Signal Performance by Efficiency Ratio Bands
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_performance_by_er AS
SELECT
    CASE
        WHEN efficiency_ratio >= 0.7 THEN 'strong_trend (0.7+)'
        WHEN efficiency_ratio >= 0.5 THEN 'moderate_trend (0.5-0.7)'
        WHEN efficiency_ratio >= 0.3 THEN 'weak_trend (0.3-0.5)'
        ELSE 'choppy (<0.3)'
    END as er_band,
    COUNT(*) as total_signals,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    ROUND(AVG(entry_quality_score)::numeric, 3) as avg_entry_quality,
    COUNT(CASE WHEN claude_approved = true THEN 1 END) as claude_approved,
    ROUND(
        COUNT(CASE WHEN claude_approved = true THEN 1 END)::numeric /
        NULLIF(COUNT(*), 0) * 100, 1
    ) as approval_rate_pct
FROM alert_history
WHERE alert_timestamp >= NOW() - INTERVAL '30 days'
  AND efficiency_ratio IS NOT NULL
GROUP BY
    CASE
        WHEN efficiency_ratio >= 0.7 THEN 'strong_trend (0.7+)'
        WHEN efficiency_ratio >= 0.5 THEN 'moderate_trend (0.5-0.7)'
        WHEN efficiency_ratio >= 0.3 THEN 'weak_trend (0.3-0.5)'
        ELSE 'choppy (<0.3)'
    END
ORDER BY
    CASE
        WHEN efficiency_ratio >= 0.7 THEN 1
        WHEN efficiency_ratio >= 0.5 THEN 2
        WHEN efficiency_ratio >= 0.3 THEN 3
        ELSE 4
    END;

COMMENT ON VIEW v_signal_performance_by_er IS
    'Signal performance metrics aggregated by Efficiency Ratio bands (last 30 days)';


-- ============================================================================
-- ANALYSIS VIEW: Rejection Analysis by Market Conditions
-- ============================================================================

CREATE OR REPLACE VIEW v_rejection_analysis_by_conditions AS
SELECT
    rejection_stage,
    market_regime_detected as regime,
    volatility_state,
    COUNT(*) as rejection_count,
    ROUND(AVG(efficiency_ratio)::numeric, 4) as avg_er,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence_at_rejection,
    ROUND(AVG(adx_value)::numeric, 1) as avg_adx
FROM smc_simple_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'
  AND market_regime_detected IS NOT NULL
GROUP BY rejection_stage, market_regime_detected, volatility_state
ORDER BY rejection_count DESC;

COMMENT ON VIEW v_rejection_analysis_by_conditions IS
    'Rejection analysis by rejection stage, market regime, and volatility (last 30 days)';


-- ============================================================================
-- DONE
-- ============================================================================
-- Run this migration with:
-- docker exec postgres psql -U postgres -d forex -f /path/to/add_performance_metrics_columns.sql
--
-- Or from within the container:
-- psql -U postgres -d forex -f /app/forex_scanner/migrations/add_performance_metrics_columns.sql
-- ============================================================================
