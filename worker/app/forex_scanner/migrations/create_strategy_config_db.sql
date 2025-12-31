-- ============================================================================
-- SMC SIMPLE STRATEGY DATABASE CONFIGURATION SCHEMA
-- ============================================================================
-- Database: strategy_config (must be created separately)
-- Purpose: Store all SMC Simple strategy configuration parameters
-- Features:
--   - Global configuration with all ~80 parameters
--   - Per-pair overrides for maximum flexibility
--   - Audit trail for change history
--   - Parameter metadata for UI rendering
-- ============================================================================

-- ============================================================================
-- TABLE 1: smc_simple_global_config
-- Stores the global (default) configuration parameters
-- ============================================================================
CREATE TABLE IF NOT EXISTS smc_simple_global_config (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL DEFAULT '2.11.0',
    is_active BOOLEAN DEFAULT TRUE,

    -- STRATEGY METADATA
    strategy_name VARCHAR(50) DEFAULT 'SMC_SIMPLE',
    strategy_status TEXT,

    -- TIER 1: 4H DIRECTIONAL BIAS
    htf_timeframe VARCHAR(10) DEFAULT '4h',
    ema_period INTEGER DEFAULT 50,
    ema_buffer_pips DECIMAL(6,2) DEFAULT 2.5,
    require_close_beyond_ema BOOLEAN DEFAULT TRUE,
    min_distance_from_ema_pips DECIMAL(6,2) DEFAULT 3.0,

    -- TIER 2: 15M ENTRY TRIGGER
    trigger_timeframe VARCHAR(10) DEFAULT '15m',
    swing_lookback_bars INTEGER DEFAULT 20,
    swing_strength_bars INTEGER DEFAULT 2,
    require_body_close_break BOOLEAN DEFAULT FALSE,
    wick_tolerance_pips DECIMAL(6,2) DEFAULT 3.0,
    volume_confirmation_enabled BOOLEAN DEFAULT TRUE,
    volume_sma_period INTEGER DEFAULT 20,
    volume_spike_multiplier DECIMAL(4,2) DEFAULT 1.2,

    -- DYNAMIC SWING LOOKBACK
    use_dynamic_swing_lookback BOOLEAN DEFAULT TRUE,
    swing_lookback_atr_low INTEGER DEFAULT 8,
    swing_lookback_atr_high INTEGER DEFAULT 15,
    swing_lookback_min INTEGER DEFAULT 15,
    swing_lookback_max INTEGER DEFAULT 30,

    -- TIER 3: 5M EXECUTION
    entry_timeframe VARCHAR(10) DEFAULT '5m',
    pullback_enabled BOOLEAN DEFAULT TRUE,
    fib_pullback_min DECIMAL(5,3) DEFAULT 0.236,
    fib_pullback_max DECIMAL(5,3) DEFAULT 0.700,
    fib_optimal_zone_min DECIMAL(5,3) DEFAULT 0.382,
    fib_optimal_zone_max DECIMAL(5,3) DEFAULT 0.618,
    max_pullback_wait_bars INTEGER DEFAULT 12,
    pullback_confirmation_bars INTEGER DEFAULT 2,

    -- MOMENTUM MODE
    momentum_mode_enabled BOOLEAN DEFAULT TRUE,
    momentum_min_depth DECIMAL(5,3) DEFAULT -0.50,
    momentum_max_depth DECIMAL(5,3) DEFAULT 0.0,
    momentum_confidence_penalty DECIMAL(4,3) DEFAULT 0.05,

    -- ATR-BASED SWING VALIDATION
    use_atr_swing_validation BOOLEAN DEFAULT TRUE,
    atr_period INTEGER DEFAULT 14,
    min_swing_atr_multiplier DECIMAL(4,3) DEFAULT 0.25,
    fallback_min_swing_pips DECIMAL(6,2) DEFAULT 5.0,

    -- MOMENTUM QUALITY FILTER
    momentum_quality_enabled BOOLEAN DEFAULT TRUE,
    min_breakout_atr_ratio DECIMAL(4,2) DEFAULT 0.5,
    min_body_percentage DECIMAL(4,2) DEFAULT 0.20,

    -- LIMIT ORDER CONFIGURATION
    limit_order_enabled BOOLEAN DEFAULT TRUE,
    limit_expiry_minutes INTEGER DEFAULT 45,
    pullback_offset_atr_factor DECIMAL(4,2) DEFAULT 0.2,
    pullback_offset_min_pips DECIMAL(5,2) DEFAULT 2.0,
    pullback_offset_max_pips DECIMAL(5,2) DEFAULT 3.0,
    momentum_offset_pips DECIMAL(5,2) DEFAULT 3.0,
    min_risk_after_offset_pips DECIMAL(5,2) DEFAULT 5.0,
    max_sl_atr_multiplier DECIMAL(4,2) DEFAULT 3.0,
    max_sl_absolute_pips DECIMAL(5,2) DEFAULT 30.0,
    max_risk_after_offset_pips DECIMAL(5,2) DEFAULT 55.0,

    -- RISK MANAGEMENT
    min_rr_ratio DECIMAL(4,2) DEFAULT 1.5,
    optimal_rr_ratio DECIMAL(4,2) DEFAULT 2.5,
    max_rr_ratio DECIMAL(4,2) DEFAULT 5.0,
    sl_buffer_pips INTEGER DEFAULT 6,
    sl_atr_multiplier DECIMAL(4,2) DEFAULT 1.0,
    use_atr_stop BOOLEAN DEFAULT TRUE,
    min_tp_pips INTEGER DEFAULT 8,
    use_swing_target BOOLEAN DEFAULT TRUE,
    tp_structure_lookback INTEGER DEFAULT 50,
    risk_per_trade_pct DECIMAL(4,2) DEFAULT 1.0,

    -- SESSION FILTER
    session_filter_enabled BOOLEAN DEFAULT TRUE,
    london_session_start TIME DEFAULT '07:00',
    london_session_end TIME DEFAULT '16:00',
    ny_session_start TIME DEFAULT '12:00',
    ny_session_end TIME DEFAULT '21:00',
    allowed_sessions TEXT[] DEFAULT ARRAY['london', 'new_york', 'overlap'],
    block_asian_session BOOLEAN DEFAULT TRUE,

    -- SIGNAL LIMITS
    max_concurrent_signals INTEGER DEFAULT 3,
    signal_cooldown_hours INTEGER DEFAULT 3,

    -- ADAPTIVE COOLDOWN (JSONB for complex nested structure)
    adaptive_cooldown_config JSONB DEFAULT '{
        "enabled": true,
        "base_cooldown_hours": 2.0,
        "cooldown_after_win_multiplier": 0.5,
        "cooldown_after_loss_multiplier": 1.5,
        "consecutive_loss_penalty_hours": 1.0,
        "max_consecutive_losses_before_block": 3,
        "consecutive_loss_block_hours": 8.0,
        "win_rate_lookback_trades": 20,
        "high_win_rate_threshold": 0.60,
        "low_win_rate_threshold": 0.40,
        "critical_win_rate_threshold": 0.30,
        "high_win_rate_cooldown_reduction": 0.25,
        "low_win_rate_cooldown_increase": 0.50,
        "high_volatility_atr_multiplier": 1.5,
        "volatility_cooldown_adjustment": 0.30,
        "strong_trend_cooldown_reduction": 0.30,
        "session_change_reset_cooldown": true,
        "min_cooldown_hours": 1.0,
        "max_cooldown_hours": 12.0
    }'::jsonb,

    -- CONFIDENCE SCORING
    min_confidence_threshold DECIMAL(4,3) DEFAULT 0.48,
    max_confidence_threshold DECIMAL(4,3) DEFAULT 0.75,
    high_confidence_threshold DECIMAL(4,3) DEFAULT 0.75,
    confidence_weights JSONB DEFAULT '{
        "ema_alignment": 0.20,
        "swing_break_quality": 0.20,
        "volume_strength": 0.20,
        "pullback_quality": 0.20,
        "rr_ratio": 0.20
    }'::jsonb,

    -- VOLUME FILTER
    min_volume_ratio DECIMAL(4,2) DEFAULT 0.50,
    volume_filter_enabled BOOLEAN DEFAULT TRUE,
    allow_no_volume_data BOOLEAN DEFAULT TRUE,

    -- DYNAMIC CONFIDENCE THRESHOLDS (v2.11.0)
    volume_adjusted_confidence_enabled BOOLEAN DEFAULT TRUE,
    high_volume_threshold DECIMAL(4,2) DEFAULT 0.70,
    atr_adjusted_confidence_enabled BOOLEAN DEFAULT TRUE,
    low_atr_threshold DECIMAL(10,6) DEFAULT 0.0004,
    high_atr_threshold DECIMAL(10,6) DEFAULT 0.0008,
    ema_distance_adjusted_confidence_enabled BOOLEAN DEFAULT TRUE,
    near_ema_threshold_pips DECIMAL(5,2) DEFAULT 20.0,
    far_ema_threshold_pips DECIMAL(5,2) DEFAULT 30.0,

    -- MACD ALIGNMENT FILTER
    macd_alignment_filter_enabled BOOLEAN DEFAULT TRUE,
    macd_alignment_mode VARCHAR(20) DEFAULT 'momentum',
    macd_min_strength DECIMAL(10,8) DEFAULT 0.0,

    -- LOGGING & DEBUG
    enable_debug_logging BOOLEAN DEFAULT TRUE,
    log_rejected_signals BOOLEAN DEFAULT TRUE,
    log_swing_detection BOOLEAN DEFAULT FALSE,
    log_ema_checks BOOLEAN DEFAULT FALSE,

    -- REJECTION TRACKING
    rejection_tracking_enabled BOOLEAN DEFAULT TRUE,
    rejection_batch_size INTEGER DEFAULT 50,
    rejection_log_to_console BOOLEAN DEFAULT FALSE,
    rejection_retention_days INTEGER DEFAULT 90,

    -- BACKTEST SETTINGS
    backtest_spread_pips DECIMAL(4,2) DEFAULT 1.5,
    backtest_slippage_pips DECIMAL(4,2) DEFAULT 0.5,

    -- ENABLED PAIRS (array)
    enabled_pairs TEXT[] DEFAULT ARRAY[
        'CS.D.EURUSD.CEEM.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP',
        'CS.D.USDCAD.MINI.IP',
        'CS.D.NZDUSD.MINI.IP',
        'CS.D.EURJPY.MINI.IP',
        'CS.D.AUDJPY.MINI.IP'
    ],

    -- PAIR PIP VALUES (JSONB)
    pair_pip_values JSONB DEFAULT '{
        "CS.D.EURUSD.CEEM.IP": 1.0,
        "CS.D.GBPUSD.MINI.IP": 0.0001,
        "CS.D.USDJPY.MINI.IP": 0.01,
        "CS.D.USDCHF.MINI.IP": 0.0001,
        "CS.D.AUDUSD.MINI.IP": 0.0001,
        "CS.D.USDCAD.MINI.IP": 0.0001,
        "CS.D.NZDUSD.MINI.IP": 0.0001,
        "CS.D.EURJPY.MINI.IP": 0.01,
        "CS.D.GBPJPY.MINI.IP": 0.01,
        "CS.D.AUDJPY.MINI.IP": 0.01
    }'::jsonb,

    -- AUDIT FIELDS
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'system',
    change_reason TEXT
);

-- ============================================================================
-- TABLE 2: smc_simple_pair_overrides
-- Per-pair parameter overrides (any parameter can be overridden)
-- ============================================================================
CREATE TABLE IF NOT EXISTS smc_simple_pair_overrides (
    id SERIAL PRIMARY KEY,
    config_id INTEGER REFERENCES smc_simple_global_config(id) ON DELETE CASCADE,
    epic VARCHAR(50) NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    description TEXT,

    -- All overridable parameters stored as JSONB for maximum flexibility
    parameter_overrides JSONB DEFAULT '{}'::jsonb,

    -- Specific confidence adjustments per condition (v2.11.0)
    high_volume_confidence DECIMAL(4,3),
    low_atr_confidence DECIMAL(4,3),
    high_atr_confidence DECIMAL(4,3),
    near_ema_confidence DECIMAL(4,3),
    far_ema_confidence DECIMAL(4,3),

    -- Session overrides
    allow_asian_session BOOLEAN,

    -- SL buffer override
    sl_buffer_pips INTEGER,

    -- Min confidence override
    min_confidence DECIMAL(4,3),

    -- MACD filter override
    macd_filter_enabled BOOLEAN,

    -- Volume ratio override
    min_volume_ratio DECIMAL(4,2),

    -- Blocking conditions (JSONB for complex rules)
    blocking_conditions JSONB DEFAULT NULL,

    -- AUDIT FIELDS
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'system',
    change_reason TEXT,

    UNIQUE(config_id, epic)
);

-- ============================================================================
-- TABLE 3: smc_simple_config_audit
-- Audit trail for all configuration changes
-- ============================================================================
CREATE TABLE IF NOT EXISTS smc_simple_config_audit (
    id SERIAL PRIMARY KEY,
    config_id INTEGER REFERENCES smc_simple_global_config(id),
    pair_override_id INTEGER REFERENCES smc_simple_pair_overrides(id),
    change_type VARCHAR(20) NOT NULL,
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    change_reason TEXT,
    previous_values JSONB,
    new_values JSONB
);

-- ============================================================================
-- TABLE 4: smc_simple_parameter_metadata
-- Parameter metadata for UI rendering
-- ============================================================================
CREATE TABLE IF NOT EXISTS smc_simple_parameter_metadata (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    data_type VARCHAR(20) NOT NULL,
    min_value DECIMAL(15,6),
    max_value DECIMAL(15,6),
    default_value TEXT,
    description TEXT,
    help_text TEXT,
    display_order INTEGER DEFAULT 0,
    is_advanced BOOLEAN DEFAULT FALSE,
    requires_restart BOOLEAN DEFAULT FALSE,
    valid_options JSONB,
    unit VARCHAR(20)
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_global_config_active
    ON smc_simple_global_config(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_pair_overrides_epic
    ON smc_simple_pair_overrides(epic);
CREATE INDEX IF NOT EXISTS idx_pair_overrides_config
    ON smc_simple_pair_overrides(config_id);
CREATE INDEX IF NOT EXISTS idx_audit_changed_at
    ON smc_simple_config_audit(changed_at DESC);
CREATE INDEX IF NOT EXISTS idx_metadata_category
    ON smc_simple_parameter_metadata(category, display_order);

-- ============================================================================
-- TRIGGER: Auto-update updated_at timestamp
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_global_config_updated_at ON smc_simple_global_config;
CREATE TRIGGER update_global_config_updated_at
    BEFORE UPDATE ON smc_simple_global_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_pair_overrides_updated_at ON smc_simple_pair_overrides;
CREATE TRIGGER update_pair_overrides_updated_at
    BEFORE UPDATE ON smc_simple_pair_overrides
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE smc_simple_global_config IS 'Global configuration parameters for SMC Simple strategy';
COMMENT ON TABLE smc_simple_pair_overrides IS 'Per-pair parameter overrides for SMC Simple strategy';
COMMENT ON TABLE smc_simple_config_audit IS 'Audit trail for configuration changes';
COMMENT ON TABLE smc_simple_parameter_metadata IS 'Parameter metadata for UI rendering';
