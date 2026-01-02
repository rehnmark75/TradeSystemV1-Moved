-- ============================================================================
-- SCANNER GLOBAL CONFIGURATION DATABASE SCHEMA
-- ============================================================================
-- Database: strategy_config (same database as SMC Simple config)
-- Purpose: Store all global scanner configuration parameters
--
-- Usage:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/create_scanner_global_config_tables.sql
-- ============================================================================

-- Ensure we're in the right database
\c strategy_config;

-- ============================================================================
-- TABLE 1: scanner_global_config
-- ============================================================================
-- Main configuration table storing all scanner global settings
-- Only one row should be active at a time (is_active = TRUE)

CREATE TABLE IF NOT EXISTS scanner_global_config (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    is_active BOOLEAN DEFAULT TRUE,

    -- =========================================================================
    -- SCANNER CORE SETTINGS
    -- =========================================================================
    scan_interval INTEGER NOT NULL DEFAULT 120,
    min_confidence DECIMAL(4,3) NOT NULL DEFAULT 0.40,
    default_timeframe VARCHAR(10) NOT NULL DEFAULT '15m',
    use_1m_base_synthesis BOOLEAN NOT NULL DEFAULT TRUE,
    scan_align_to_boundaries BOOLEAN NOT NULL DEFAULT TRUE,
    scan_boundary_offset_seconds INTEGER NOT NULL DEFAULT 60,

    -- =========================================================================
    -- DUPLICATE DETECTION SETTINGS
    -- =========================================================================
    enable_duplicate_check BOOLEAN NOT NULL DEFAULT TRUE,
    duplicate_sensitivity VARCHAR(20) NOT NULL DEFAULT 'smart',
    signal_cooldown_minutes INTEGER NOT NULL DEFAULT 15,
    alert_cooldown_minutes INTEGER NOT NULL DEFAULT 5,
    strategy_cooldown_minutes INTEGER NOT NULL DEFAULT 3,
    global_cooldown_seconds INTEGER NOT NULL DEFAULT 30,
    max_alerts_per_hour INTEGER NOT NULL DEFAULT 50,
    max_alerts_per_epic_hour INTEGER NOT NULL DEFAULT 6,
    price_similarity_threshold DECIMAL(8,6) NOT NULL DEFAULT 0.0002,
    confidence_similarity_threshold DECIMAL(4,3) NOT NULL DEFAULT 0.05,
    deduplication_preset VARCHAR(20) NOT NULL DEFAULT 'standard',
    use_database_dedup_check BOOLEAN NOT NULL DEFAULT TRUE,
    database_dedup_window_minutes INTEGER NOT NULL DEFAULT 15,
    enable_signal_hash_check BOOLEAN NOT NULL DEFAULT FALSE,
    deduplication_debug_mode BOOLEAN NOT NULL DEFAULT FALSE,
    enable_price_similarity_check BOOLEAN NOT NULL DEFAULT TRUE,
    enable_strategy_cooldowns BOOLEAN NOT NULL DEFAULT TRUE,
    deduplication_lookback_hours INTEGER NOT NULL DEFAULT 2,

    -- =========================================================================
    -- RISK MANAGEMENT SETTINGS
    -- =========================================================================
    position_size_percent DECIMAL(4,2) NOT NULL DEFAULT 1.0,
    stop_loss_pips INTEGER NOT NULL DEFAULT 5,
    take_profit_pips INTEGER NOT NULL DEFAULT 15,
    max_open_positions INTEGER NOT NULL DEFAULT 3,
    max_daily_trades INTEGER NOT NULL DEFAULT 10,
    risk_per_trade_percent DECIMAL(5,4) NOT NULL DEFAULT 0.02,
    min_position_size DECIMAL(5,3) NOT NULL DEFAULT 0.01,
    max_position_size DECIMAL(5,2) NOT NULL DEFAULT 1.0,
    max_risk_per_trade INTEGER NOT NULL DEFAULT 30,
    default_risk_reward DECIMAL(4,2) NOT NULL DEFAULT 2.0,
    default_stop_distance INTEGER NOT NULL DEFAULT 20,

    -- =========================================================================
    -- TRADING HOURS SETTINGS
    -- =========================================================================
    trading_start_hour INTEGER NOT NULL DEFAULT 0,
    trading_end_hour INTEGER NOT NULL DEFAULT 23,
    respect_market_hours BOOLEAN NOT NULL DEFAULT FALSE,
    weekend_scanning BOOLEAN NOT NULL DEFAULT FALSE,
    enable_trading_time_controls BOOLEAN NOT NULL DEFAULT TRUE,
    trading_cutoff_time_utc INTEGER NOT NULL DEFAULT 20,
    trade_cooldown_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    trade_cooldown_minutes INTEGER NOT NULL DEFAULT 30,
    user_timezone VARCHAR(50) NOT NULL DEFAULT 'Europe/Stockholm',
    respect_trading_hours BOOLEAN NOT NULL DEFAULT FALSE,

    -- =========================================================================
    -- SAFETY FILTER SETTINGS
    -- =========================================================================
    enable_critical_safety_filters BOOLEAN NOT NULL DEFAULT TRUE,
    enable_ema200_contradiction_filter BOOLEAN NOT NULL DEFAULT TRUE,
    enable_ema_stack_contradiction_filter BOOLEAN NOT NULL DEFAULT TRUE,
    require_indicator_consensus BOOLEAN NOT NULL DEFAULT TRUE,
    min_confirming_indicators INTEGER NOT NULL DEFAULT 1,
    enable_emergency_circuit_breaker BOOLEAN NOT NULL DEFAULT TRUE,
    max_contradictions_allowed INTEGER NOT NULL DEFAULT 5,
    active_safety_preset VARCHAR(20) NOT NULL DEFAULT 'balanced',
    enable_large_candle_filter BOOLEAN NOT NULL DEFAULT TRUE,
    large_candle_atr_multiplier DECIMAL(4,2) NOT NULL DEFAULT 2.5,
    consecutive_large_candles_threshold INTEGER NOT NULL DEFAULT 2,
    movement_lookback_periods INTEGER NOT NULL DEFAULT 3,
    large_candle_filter_cooldown INTEGER NOT NULL DEFAULT 3,
    ema200_minimum_margin DECIMAL(6,4) NOT NULL DEFAULT 0.002,
    safety_filter_log_level VARCHAR(20) NOT NULL DEFAULT 'ERROR',
    excessive_movement_threshold_pips INTEGER NOT NULL DEFAULT 15,

    -- =========================================================================
    -- ADX FILTER SETTINGS
    -- =========================================================================
    adx_filter_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    adx_filter_mode VARCHAR(20) NOT NULL DEFAULT 'moderate',
    adx_period INTEGER NOT NULL DEFAULT 14,
    adx_grace_period_bars INTEGER NOT NULL DEFAULT 2,

    -- ADX thresholds as JSONB for flexibility
    adx_thresholds JSONB NOT NULL DEFAULT '{
        "STRONG_TREND": 25.0,
        "MODERATE_TREND": 22.0,
        "WEAK_TREND": 15.0,
        "VERY_WEAK": 10.0
    }'::jsonb,

    -- ADX pair-specific multipliers
    adx_pair_multipliers JSONB NOT NULL DEFAULT '{
        "EURUSD": 1.0,
        "GBPUSD": 0.9,
        "USDJPY": 1.1,
        "EURJPY": 0.85,
        "GBPJPY": 0.8,
        "USDCHF": 1.0,
        "DEFAULT": 1.0
    }'::jsonb,

    -- =========================================================================
    -- DEDUPLICATION PRESETS (JSONB)
    -- =========================================================================
    deduplication_presets JSONB NOT NULL DEFAULT '{
        "strict": {
            "alert_cooldown_minutes": 10,
            "strategy_cooldown_minutes": 5,
            "max_alerts_per_hour": 30,
            "max_alerts_per_epic_hour": 3,
            "price_similarity_threshold": 0.0005,
            "confidence_similarity_threshold": 0.02
        },
        "standard": {
            "alert_cooldown_minutes": 5,
            "strategy_cooldown_minutes": 3,
            "max_alerts_per_hour": 50,
            "max_alerts_per_epic_hour": 6,
            "price_similarity_threshold": 0.0002,
            "confidence_similarity_threshold": 0.05
        },
        "relaxed": {
            "alert_cooldown_minutes": 2,
            "strategy_cooldown_minutes": 1,
            "max_alerts_per_hour": 100,
            "max_alerts_per_epic_hour": 12,
            "price_similarity_threshold": 0.0001,
            "confidence_similarity_threshold": 0.1
        }
    }'::jsonb,

    -- =========================================================================
    -- SAFETY FILTER PRESETS (JSONB)
    -- =========================================================================
    safety_filter_presets JSONB NOT NULL DEFAULT '{
        "strict": {
            "enable_ema200_contradiction_filter": true,
            "enable_ema_stack_contradiction_filter": true,
            "require_indicator_consensus": true,
            "max_contradictions_allowed": 0,
            "ema200_minimum_margin": 0.001
        },
        "balanced": {
            "enable_ema200_contradiction_filter": true,
            "enable_ema_stack_contradiction_filter": true,
            "require_indicator_consensus": true,
            "max_contradictions_allowed": 5,
            "ema200_minimum_margin": 0.002
        },
        "permissive": {
            "enable_ema200_contradiction_filter": true,
            "enable_ema_stack_contradiction_filter": false,
            "require_indicator_consensus": false,
            "max_contradictions_allowed": 2,
            "ema200_minimum_margin": 0.005
        },
        "emergency": {
            "enable_ema200_contradiction_filter": true,
            "enable_ema_stack_contradiction_filter": false,
            "require_indicator_consensus": false,
            "max_contradictions_allowed": 3,
            "ema200_minimum_margin": 0.01
        }
    }'::jsonb,

    -- =========================================================================
    -- LARGE CANDLE FILTER PRESETS (JSONB)
    -- =========================================================================
    large_candle_filter_presets JSONB NOT NULL DEFAULT '{
        "strict": {
            "large_candle_atr_multiplier": 2.0,
            "consecutive_large_candles_threshold": 1,
            "excessive_movement_threshold_pips": 10,
            "large_candle_filter_cooldown": 5
        },
        "balanced": {
            "large_candle_atr_multiplier": 2.5,
            "consecutive_large_candles_threshold": 2,
            "excessive_movement_threshold_pips": 15,
            "large_candle_filter_cooldown": 3
        },
        "permissive": {
            "large_candle_atr_multiplier": 3.0,
            "consecutive_large_candles_threshold": 3,
            "excessive_movement_threshold_pips": 20,
            "large_candle_filter_cooldown": 2
        }
    }'::jsonb,

    -- =========================================================================
    -- AUDIT FIELDS
    -- =========================================================================
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) NOT NULL DEFAULT 'system',
    change_reason TEXT
);

-- ============================================================================
-- TABLE 2: scanner_config_audit
-- ============================================================================
-- Audit trail for all configuration changes

CREATE TABLE IF NOT EXISTS scanner_config_audit (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL REFERENCES scanner_global_config(id) ON DELETE CASCADE,
    change_type VARCHAR(20) NOT NULL,  -- 'UPDATE', 'INSERT', 'PRESET_APPLY'
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    change_reason TEXT,
    previous_values JSONB,
    new_values JSONB,
    category VARCHAR(50)  -- 'core', 'dedup', 'risk', 'trading_hours', 'safety', 'adx'
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_scanner_config_active
    ON scanner_global_config(is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_scanner_audit_changed_at
    ON scanner_config_audit(changed_at DESC);

CREATE INDEX IF NOT EXISTS idx_scanner_audit_category
    ON scanner_config_audit(category);

CREATE INDEX IF NOT EXISTS idx_scanner_audit_config_id
    ON scanner_config_audit(config_id);

-- ============================================================================
-- TRIGGER: Auto-update updated_at timestamp
-- ============================================================================
-- First check if the function exists (it should from SMC Simple migration)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column'
    ) THEN
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $trigger$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $trigger$ LANGUAGE plpgsql;
    END IF;
END
$$;

DROP TRIGGER IF EXISTS update_scanner_config_updated_at ON scanner_global_config;
CREATE TRIGGER update_scanner_config_updated_at
    BEFORE UPDATE ON scanner_global_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA: Insert initial config with current production values
-- ============================================================================
-- Only insert if no active config exists
INSERT INTO scanner_global_config (
    version,
    is_active,
    updated_by,
    change_reason,

    -- Core settings (from config.py current values)
    scan_interval,
    min_confidence,
    default_timeframe,
    use_1m_base_synthesis,
    scan_align_to_boundaries,
    scan_boundary_offset_seconds,

    -- Dedup settings
    enable_duplicate_check,
    duplicate_sensitivity,
    signal_cooldown_minutes,
    alert_cooldown_minutes,
    strategy_cooldown_minutes,
    global_cooldown_seconds,
    max_alerts_per_hour,
    max_alerts_per_epic_hour,
    deduplication_preset,

    -- Risk settings
    position_size_percent,
    stop_loss_pips,
    take_profit_pips,
    max_open_positions,
    max_daily_trades,

    -- Trading hours
    trading_start_hour,
    trading_end_hour,
    respect_market_hours,
    weekend_scanning,
    trading_cutoff_time_utc,
    trade_cooldown_minutes,

    -- Safety filters
    enable_critical_safety_filters,
    active_safety_preset,
    max_contradictions_allowed,

    -- ADX
    adx_filter_enabled,
    adx_filter_mode
)
SELECT
    '1.0.0',
    TRUE,
    'migration',
    'Initial migration from config.py',

    -- Core
    120,      -- scan_interval
    0.40,     -- min_confidence
    '15m',    -- default_timeframe
    TRUE,     -- use_1m_base_synthesis
    TRUE,     -- scan_align_to_boundaries
    60,       -- scan_boundary_offset_seconds

    -- Dedup
    TRUE,     -- enable_duplicate_check
    'smart',  -- duplicate_sensitivity
    15,       -- signal_cooldown_minutes
    5,        -- alert_cooldown_minutes
    3,        -- strategy_cooldown_minutes
    30,       -- global_cooldown_seconds
    50,       -- max_alerts_per_hour
    6,        -- max_alerts_per_epic_hour
    'standard', -- deduplication_preset

    -- Risk
    1.0,      -- position_size_percent
    5,        -- stop_loss_pips
    15,       -- take_profit_pips
    3,        -- max_open_positions
    10,       -- max_daily_trades

    -- Trading hours
    0,        -- trading_start_hour
    23,       -- trading_end_hour
    FALSE,    -- respect_market_hours
    FALSE,    -- weekend_scanning
    20,       -- trading_cutoff_time_utc
    30,       -- trade_cooldown_minutes

    -- Safety
    TRUE,     -- enable_critical_safety_filters
    'balanced', -- active_safety_preset
    5,        -- max_contradictions_allowed

    -- ADX
    FALSE,    -- adx_filter_enabled
    'moderate' -- adx_filter_mode
WHERE NOT EXISTS (
    SELECT 1 FROM scanner_global_config WHERE is_active = TRUE
);

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE scanner_global_config IS 'Global configuration parameters for Forex Scanner - database is the only source of truth';
COMMENT ON TABLE scanner_config_audit IS 'Audit trail for scanner configuration changes';

COMMENT ON COLUMN scanner_global_config.version IS 'Configuration version for tracking changes';
COMMENT ON COLUMN scanner_global_config.is_active IS 'Only one config should be active at a time';
COMMENT ON COLUMN scanner_global_config.scan_interval IS 'Seconds between scanner runs';
COMMENT ON COLUMN scanner_global_config.min_confidence IS 'Minimum signal confidence threshold (0.0-1.0)';
COMMENT ON COLUMN scanner_global_config.deduplication_preset IS 'Active preset: strict, standard, relaxed';
COMMENT ON COLUMN scanner_global_config.active_safety_preset IS 'Active preset: strict, balanced, permissive, emergency';

-- ============================================================================
-- VERIFICATION
-- ============================================================================
DO $$
DECLARE
    config_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO config_count FROM scanner_global_config WHERE is_active = TRUE;

    IF config_count = 1 THEN
        RAISE NOTICE 'SUCCESS: Scanner global config table created with 1 active configuration';
    ELSIF config_count = 0 THEN
        RAISE WARNING 'No active configuration found - insert seed data manually';
    ELSE
        RAISE WARNING 'Multiple active configurations found (%) - only one should be active', config_count;
    END IF;
END
$$;

-- Show the created config
SELECT
    id,
    version,
    is_active,
    scan_interval,
    min_confidence,
    deduplication_preset,
    active_safety_preset,
    updated_by,
    created_at
FROM scanner_global_config
WHERE is_active = TRUE;
