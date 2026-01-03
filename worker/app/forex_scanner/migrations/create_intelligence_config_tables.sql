-- ============================================================================
-- INTELLIGENCE SYSTEM DATABASE CONFIGURATION SCHEMA
-- ============================================================================
-- Database: strategy_config (same database as SMC Simple config)
-- Purpose: Store all Market Intelligence configuration parameters
-- Features:
--   - Global intelligence configuration with all parameters
--   - Regime-Strategy confidence modifiers
--   - Preset definitions
--   - Audit trail for change history
--
-- Migration created: 2026-01-03
-- Based on: configdata/market_intelligence_config.py (LEGACY after migration)
-- ============================================================================

-- ============================================================================
-- TABLE 1: intelligence_global_config
-- Stores all global intelligence parameters (replaces market_intelligence_config.py)
-- ============================================================================
CREATE TABLE IF NOT EXISTS intelligence_global_config (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(100) NOT NULL UNIQUE,
    parameter_value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    subcategory VARCHAR(50),
    display_order INTEGER DEFAULT 0,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_editable BOOLEAN DEFAULT TRUE,
    min_value DECIMAL(15,6),
    max_value DECIMAL(15,6),
    valid_options JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- TABLE 2: intelligence_presets
-- Preset definitions (disabled, minimal, balanced, conservative, collect_only)
-- ============================================================================
CREATE TABLE IF NOT EXISTS intelligence_presets (
    id SERIAL PRIMARY KEY,
    preset_name VARCHAR(50) NOT NULL UNIQUE,
    threshold DECIMAL(4,3) NOT NULL DEFAULT 0.0,
    use_intelligence_engine BOOLEAN NOT NULL DEFAULT FALSE,
    description TEXT,
    display_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- TABLE 3: intelligence_preset_components
-- Component enablement per preset
-- ============================================================================
CREATE TABLE IF NOT EXISTS intelligence_preset_components (
    id SERIAL PRIMARY KEY,
    preset_id INTEGER REFERENCES intelligence_presets(id) ON DELETE CASCADE,
    component_name VARCHAR(50) NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(preset_id, component_name)
);

-- ============================================================================
-- TABLE 4: intelligence_regime_modifiers
-- Regime-Strategy confidence modifiers (probabilistic system)
-- ============================================================================
CREATE TABLE IF NOT EXISTS intelligence_regime_modifiers (
    id SERIAL PRIMARY KEY,
    regime VARCHAR(30) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    confidence_modifier DECIMAL(4,3) NOT NULL DEFAULT 1.0,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(regime, strategy)
);

-- ============================================================================
-- TABLE 5: intelligence_config_audit
-- Audit trail for all intelligence configuration changes
-- ============================================================================
CREATE TABLE IF NOT EXISTS intelligence_config_audit (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id INTEGER,
    change_type VARCHAR(20) NOT NULL,
    changed_by VARCHAR(100) NOT NULL DEFAULT 'system',
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    change_reason TEXT,
    previous_values JSONB,
    new_values JSONB
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_intel_config_category ON intelligence_global_config(category, display_order);
CREATE INDEX IF NOT EXISTS idx_intel_config_active ON intelligence_global_config(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_intel_presets_active ON intelligence_presets(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_intel_regime_modifiers ON intelligence_regime_modifiers(regime, strategy);
CREATE INDEX IF NOT EXISTS idx_intel_audit_changed_at ON intelligence_config_audit(changed_at DESC);

-- ============================================================================
-- TRIGGER: Auto-update updated_at timestamp
-- ============================================================================
-- Reuse existing function if it exists, otherwise create
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_intel_config_updated_at ON intelligence_global_config;
CREATE TRIGGER update_intel_config_updated_at
    BEFORE UPDATE ON intelligence_global_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_intel_presets_updated_at ON intelligence_presets;
CREATE TRIGGER update_intel_presets_updated_at
    BEFORE UPDATE ON intelligence_presets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_intel_regime_modifiers_updated_at ON intelligence_regime_modifiers;
CREATE TRIGGER update_intel_regime_modifiers_updated_at
    BEFORE UPDATE ON intelligence_regime_modifiers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA: Global Intelligence Configuration
-- ============================================================================
INSERT INTO intelligence_global_config (parameter_name, parameter_value, value_type, category, subcategory, description, display_order, min_value, max_value, valid_options) VALUES
    -- Core Settings
    ('intelligence_preset', 'collect_only', 'string', 'core', NULL, 'Active preset: disabled, minimal, balanced, conservative, collect_only', 1, NULL, NULL, '["disabled", "minimal", "balanced", "conservative", "collect_only", "testing"]'),
    ('intelligence_mode', 'live_only', 'string', 'core', NULL, 'Mode: disabled, backtest_consistent, live_only, enhanced', 2, NULL, NULL, '["disabled", "backtest_consistent", "live_only", "enhanced"]'),
    ('enable_market_intelligence', 'true', 'bool', 'core', NULL, 'Master switch for market intelligence system', 3, NULL, NULL, NULL),

    -- Component Weights
    ('weight_market_regime', '0.25', 'float', 'weights', NULL, 'Weight for market regime component in scoring', 10, 0.0, 1.0, NULL),
    ('weight_volatility', '0.25', 'float', 'weights', NULL, 'Weight for volatility component in scoring', 11, 0.0, 1.0, NULL),
    ('weight_volume', '0.25', 'float', 'weights', NULL, 'Weight for volume component in scoring', 12, 0.0, 1.0, NULL),
    ('weight_time', '0.0', 'float', 'weights', NULL, 'Weight for time-of-day component (disabled by default)', 13, 0.0, 1.0, NULL),
    ('weight_confidence', '0.25', 'float', 'weights', NULL, 'Weight for confidence component in scoring', 14, 0.0, 1.0, NULL),

    -- Component Enablement (defaults for collect_only mode - no filtering)
    ('enable_market_regime_filter', 'false', 'bool', 'components', NULL, 'Enable market regime filtering (blocks signals in unsuitable regimes)', 20, NULL, NULL, NULL),
    ('enable_volatility_filter', 'false', 'bool', 'components', NULL, 'Enable volatility filtering', 21, NULL, NULL, NULL),
    ('enable_volume_filter', 'false', 'bool', 'components', NULL, 'Enable volume filtering', 22, NULL, NULL, NULL),
    ('enable_time_filter', 'false', 'bool', 'components', NULL, 'Enable time-of-day filtering', 23, NULL, NULL, NULL),
    ('enable_confidence_filter', 'false', 'bool', 'components', NULL, 'Enable confidence filtering', 24, NULL, NULL, NULL),
    ('enable_spread_filter', 'false', 'bool', 'components', NULL, 'Enable spread filtering', 25, NULL, NULL, NULL),
    ('enable_recent_signals_filter', 'true', 'bool', 'components', NULL, 'Enable recent signals deduplication filter', 26, NULL, NULL, NULL),

    -- Smart Money Settings
    ('enable_smart_money_collection', 'true', 'bool', 'smart_money', NULL, 'Collect Smart Money structure data (BOS/ChoCh/Swing)', 30, NULL, NULL, NULL),
    ('enable_order_flow_collection', 'true', 'bool', 'smart_money', NULL, 'Collect Order Flow data (Order Blocks/FVGs)', 31, NULL, NULL, NULL),
    ('smart_money_structure_validation', 'false', 'bool', 'smart_money', NULL, 'Validate signals against market structure (blocks if misaligned)', 32, NULL, NULL, NULL),
    ('smart_money_order_flow_validation', 'false', 'bool', 'smart_money', NULL, 'Validate signals against order flow (blocks if misaligned)', 33, NULL, NULL, NULL),

    -- Enhanced Regime Detection (ADX-based)
    ('enhanced_regime_detection_enabled', 'true', 'bool', 'regime', NULL, 'Enable ADX-based enhanced regime detection', 40, NULL, NULL, NULL),
    ('adx_trending_threshold', '25', 'int', 'regime', NULL, 'ADX threshold for trending market detection', 41, 10, 50, NULL),
    ('adx_strong_trend_threshold', '40', 'int', 'regime', NULL, 'ADX threshold for strong trend detection', 42, 25, 60, NULL),
    ('adx_weak_trend_threshold', '20', 'int', 'regime', NULL, 'ADX threshold for weak/ranging market', 43, 10, 30, NULL),
    ('ema_alignment_weight', '0.4', 'float', 'regime', NULL, 'EMA alignment weight in trending score', 44, 0.0, 1.0, NULL),
    ('adx_weight', '0.4', 'float', 'regime', NULL, 'ADX weight in trending score', 45, 0.0, 1.0, NULL),
    ('momentum_weight', '0.2', 'float', 'regime', NULL, 'Momentum weight in trending score', 46, 0.0, 1.0, NULL),
    ('trending_score_threshold', '0.55', 'float', 'regime', NULL, 'Score threshold for trending classification', 47, 0.3, 0.8, NULL),
    ('ranging_score_threshold', '0.55', 'float', 'regime', NULL, 'Score threshold for ranging classification', 48, 0.3, 0.8, NULL),
    ('breakout_score_threshold', '0.60', 'float', 'regime', NULL, 'Score threshold for breakout potential', 49, 0.4, 0.9, NULL),
    ('collect_enhanced_regime_data', 'true', 'bool', 'regime', NULL, 'Collect enhanced regime data for analysis', 50, NULL, NULL, NULL),
    ('log_regime_comparison', 'true', 'bool', 'regime', NULL, 'Log old vs new regime detection comparison', 51, NULL, NULL, NULL),
    ('separate_volatility_from_structure', 'true', 'bool', 'regime', NULL, 'Treat volatility as orthogonal to structure', 52, NULL, NULL, NULL),
    ('volatility_as_modifier', 'true', 'bool', 'regime', NULL, 'Use volatility to modify rather than classify', 53, NULL, NULL, NULL),

    -- Probabilistic Confidence Modifiers
    ('enable_probabilistic_confidence_modifiers', 'true', 'bool', 'modifiers', NULL, 'Enable regime-strategy confidence modifier system', 60, NULL, NULL, NULL),
    ('min_confidence_modifier', '0.5', 'float', 'modifiers', NULL, 'Minimum confidence modifier (signals below this blocked)', 61, 0.1, 1.0, NULL),

    -- Market Bias Filter
    ('market_bias_filter_enabled', 'false', 'bool', 'filtering', NULL, 'Block counter-trend trades when directional consensus high', 70, NULL, NULL, NULL),
    ('market_bias_min_consensus', '0.70', 'float', 'filtering', NULL, 'Minimum directional consensus to trigger blocking', 71, 0.5, 0.95, NULL),
    ('market_intelligence_min_confidence', '0.45', 'float', 'filtering', NULL, 'Minimum intelligence confidence for signals', 72, 0.0, 1.0, NULL),
    ('block_unsuitable_regimes', 'false', 'bool', 'filtering', NULL, 'Block signals in unsuitable market regimes', 73, NULL, NULL, NULL),

    -- Storage Settings
    ('enable_intelligence_storage', 'true', 'bool', 'storage', NULL, 'Store market intelligence data per scan', 80, NULL, NULL, NULL),
    ('intelligence_cleanup_days', '60', 'int', 'storage', NULL, 'Days to keep intelligence records (auto-cleanup)', 81, 7, 365, NULL),

    -- Analysis Settings
    ('force_intelligence_analysis', 'true', 'bool', 'analysis', NULL, 'Force analysis even when market is closed', 90, NULL, NULL, NULL),
    ('intelligence_override_market_hours', 'true', 'bool', 'analysis', NULL, 'Override market hours for intelligence collection', 91, NULL, NULL, NULL),
    ('use_historical_data_for_intelligence', 'true', 'bool', 'analysis', NULL, 'Use historical data for intelligence analysis', 92, NULL, NULL, NULL),
    ('intelligence_confidence_threshold', '0.3', 'float', 'analysis', NULL, 'Base confidence threshold', 93, 0.0, 1.0, NULL),
    ('intelligence_volume_threshold', '0.2', 'float', 'analysis', NULL, 'Minimum volume threshold', 94, 0.0, 1.0, NULL),
    ('intelligence_volatility_min', '0.1', 'float', 'analysis', NULL, 'Minimum volatility threshold', 95, 0.0, 1.0, NULL),
    ('intelligence_allow_low_volatility', 'true', 'bool', 'analysis', NULL, 'Allow signals in low volatility conditions', 96, NULL, NULL, NULL),
    ('intelligence_allow_ranging_markets', 'true', 'bool', 'analysis', NULL, 'Allow signals in ranging markets', 97, NULL, NULL, NULL),

    -- Live Scanner Settings
    ('live_scanner_lookback_hours', '24', 'int', 'scanner', NULL, 'Hours of historical data for live scanner', 100, 1, 168, NULL),
    ('enable_recent_historical_scan', 'true', 'bool', 'scanner', NULL, 'Use recent historical data for scans', 101, NULL, NULL, NULL),
    ('use_backtest_data_logic_for_live', 'true', 'bool', 'scanner', NULL, 'Use backtest data logic in live mode', 102, NULL, NULL, NULL),
    ('force_market_open', 'false', 'bool', 'scanner', NULL, 'Force market open status (testing only)', 103, NULL, NULL, NULL),
    ('enable_weekend_scanning', 'false', 'bool', 'scanner', NULL, 'Allow scanning during weekends', 104, NULL, NULL, NULL),
    ('max_data_age_minutes', '60', 'int', 'scanner', NULL, 'Maximum age of data in minutes', 105, 5, 1440, NULL),
    ('minimum_candles_for_live_scan', '50', 'int', 'scanner', NULL, 'Minimum candles required for live scan', 106, 10, 500, NULL),

    -- Debug Settings
    ('intelligence_debug_mode', 'false', 'bool', 'debug', NULL, 'Enable verbose intelligence logging', 110, NULL, NULL, NULL),
    ('intelligence_log_rejections', 'true', 'bool', 'debug', NULL, 'Log rejected signal details', 111, NULL, NULL, NULL),

    -- Claude AI Integration
    ('claude_integrate_with_intelligence', 'true', 'bool', 'claude', NULL, 'Integrate intelligence data with Claude AI analysis', 120, NULL, NULL, NULL),
    ('claude_use_intelligence_in_prompts', 'true', 'bool', 'claude', NULL, 'Include intelligence data in Claude prompts', 121, NULL, NULL, NULL),
    ('claude_override_intelligence', 'true', 'bool', 'claude', NULL, 'Allow Claude to override intelligence filtering', 122, NULL, NULL, NULL),
    ('intelligence_claude_weight', '0.3', 'float', 'claude', NULL, 'Weight of Claude vs intelligence decisions', 123, 0.0, 1.0, NULL)

ON CONFLICT (parameter_name) DO UPDATE SET
    parameter_value = EXCLUDED.parameter_value,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================================
-- SEED DATA: Intelligence Presets
-- ============================================================================
INSERT INTO intelligence_presets (preset_name, threshold, use_intelligence_engine, description, display_order) VALUES
    ('disabled', 0.0, FALSE, 'No intelligence filtering - strategy signals only', 1),
    ('minimal', 0.3, FALSE, 'Minimal filtering - volume + confidence only', 2),
    ('balanced', 0.5, TRUE, 'Balanced filtering with market intelligence', 3),
    ('conservative', 0.7, TRUE, 'Conservative filtering - fewer but higher quality signals', 4),
    ('collect_only', 0.0, TRUE, 'Full engine running for data collection, but no signal filtering', 5),
    ('testing', 0.4, FALSE, 'Consistent with backtesting environment', 6)
ON CONFLICT (preset_name) DO UPDATE SET
    threshold = EXCLUDED.threshold,
    use_intelligence_engine = EXCLUDED.use_intelligence_engine,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================================
-- SEED DATA: Preset Components (for each preset)
-- ============================================================================
-- First get preset IDs
DO $$
DECLARE
    v_disabled_id INTEGER;
    v_minimal_id INTEGER;
    v_balanced_id INTEGER;
    v_conservative_id INTEGER;
    v_collect_only_id INTEGER;
    v_testing_id INTEGER;
BEGIN
    SELECT id INTO v_disabled_id FROM intelligence_presets WHERE preset_name = 'disabled';
    SELECT id INTO v_minimal_id FROM intelligence_presets WHERE preset_name = 'minimal';
    SELECT id INTO v_balanced_id FROM intelligence_presets WHERE preset_name = 'balanced';
    SELECT id INTO v_conservative_id FROM intelligence_presets WHERE preset_name = 'conservative';
    SELECT id INTO v_collect_only_id FROM intelligence_presets WHERE preset_name = 'collect_only';
    SELECT id INTO v_testing_id FROM intelligence_presets WHERE preset_name = 'testing';

    -- Disabled preset components (all disabled except confidence)
    INSERT INTO intelligence_preset_components (preset_id, component_name, is_enabled) VALUES
        (v_disabled_id, 'market_regime_filter', FALSE),
        (v_disabled_id, 'volatility_filter', FALSE),
        (v_disabled_id, 'volume_filter', FALSE),
        (v_disabled_id, 'time_filter', FALSE),
        (v_disabled_id, 'confidence_filter', TRUE)
    ON CONFLICT (preset_id, component_name) DO UPDATE SET is_enabled = EXCLUDED.is_enabled;

    -- Minimal preset components
    INSERT INTO intelligence_preset_components (preset_id, component_name, is_enabled) VALUES
        (v_minimal_id, 'market_regime_filter', FALSE),
        (v_minimal_id, 'volatility_filter', FALSE),
        (v_minimal_id, 'volume_filter', TRUE),
        (v_minimal_id, 'time_filter', FALSE),
        (v_minimal_id, 'confidence_filter', TRUE)
    ON CONFLICT (preset_id, component_name) DO UPDATE SET is_enabled = EXCLUDED.is_enabled;

    -- Balanced preset components
    INSERT INTO intelligence_preset_components (preset_id, component_name, is_enabled) VALUES
        (v_balanced_id, 'market_regime_filter', TRUE),
        (v_balanced_id, 'volatility_filter', TRUE),
        (v_balanced_id, 'volume_filter', TRUE),
        (v_balanced_id, 'time_filter', FALSE),
        (v_balanced_id, 'confidence_filter', TRUE)
    ON CONFLICT (preset_id, component_name) DO UPDATE SET is_enabled = EXCLUDED.is_enabled;

    -- Conservative preset components (all enabled)
    INSERT INTO intelligence_preset_components (preset_id, component_name, is_enabled) VALUES
        (v_conservative_id, 'market_regime_filter', TRUE),
        (v_conservative_id, 'volatility_filter', TRUE),
        (v_conservative_id, 'volume_filter', TRUE),
        (v_conservative_id, 'time_filter', TRUE),
        (v_conservative_id, 'confidence_filter', TRUE)
    ON CONFLICT (preset_id, component_name) DO UPDATE SET is_enabled = EXCLUDED.is_enabled;

    -- Collect_only preset components (all disabled for filtering, but engine runs)
    INSERT INTO intelligence_preset_components (preset_id, component_name, is_enabled) VALUES
        (v_collect_only_id, 'market_regime_filter', FALSE),
        (v_collect_only_id, 'volatility_filter', FALSE),
        (v_collect_only_id, 'volume_filter', FALSE),
        (v_collect_only_id, 'time_filter', FALSE),
        (v_collect_only_id, 'confidence_filter', FALSE)
    ON CONFLICT (preset_id, component_name) DO UPDATE SET is_enabled = EXCLUDED.is_enabled;

    -- Testing preset components
    INSERT INTO intelligence_preset_components (preset_id, component_name, is_enabled) VALUES
        (v_testing_id, 'market_regime_filter', FALSE),
        (v_testing_id, 'volatility_filter', TRUE),
        (v_testing_id, 'volume_filter', TRUE),
        (v_testing_id, 'time_filter', FALSE),
        (v_testing_id, 'confidence_filter', TRUE)
    ON CONFLICT (preset_id, component_name) DO UPDATE SET is_enabled = EXCLUDED.is_enabled;
END $$;

-- ============================================================================
-- SEED DATA: Regime-Strategy Confidence Modifiers
-- ============================================================================
INSERT INTO intelligence_regime_modifiers (regime, strategy, confidence_modifier, description) VALUES
    -- TRENDING REGIME
    ('trending', 'ema', 1.0, 'PERFECT - EMA is THE trend indicator'),
    ('trending', 'smart_money_ema', 1.0, 'PERFECT - EMA with smart money'),
    ('trending', 'ema_double', 1.0, 'PERFECT - EMA with crossover confirmation + FVG + ADX'),
    ('trending', 'macd', 1.0, 'Perfect - MACD follows trends excellently'),
    ('trending', 'smart_money_macd', 0.95, 'Excellent - MACD with smart money'),
    ('trending', 'ichimoku', 1.0, 'Perfect - cloud trend analysis'),
    ('trending', 'kama', 1.0, 'Perfect - adaptive moving average'),
    ('trending', 'momentum', 0.95, 'Excellent - momentum follows trends'),
    ('trending', 'zero_lag', 0.95, 'Excellent - fast trend response'),
    ('trending', 'bb_supertrend', 0.9, 'Very good - combines trend + volatility'),
    ('trending', 'smc', 0.7, 'Good - smart money can detect trends'),
    ('trending', 'smc_simple', 0.9, 'Excellent - SMC good in trends'),
    ('trending', 'mean_reversion', 0.3, 'Poor - fights the trend'),
    ('trending', 'ranging_market', 0.25, 'Very poor - designed for ranges'),
    ('trending', 'bollinger', 0.5, 'Moderate - can work with trend breakouts'),
    ('trending', 'scalping', 0.4, 'Poor - trends too strong for scalping'),

    -- RANGING REGIME
    ('ranging', 'ema', 0.8, 'GOOD - defines range boundaries effectively'),
    ('ranging', 'smart_money_ema', 0.85, 'Good - EMA with smart money context'),
    ('ranging', 'ema_double', 0.6, 'MODERATE - ADX filter helps avoid ranging markets'),
    ('ranging', 'macd', 0.4, 'POOR - momentum crossovers generate false signals'),
    ('ranging', 'smart_money_macd', 0.45, 'POOR - slightly better with smart money'),
    ('ranging', 'ichimoku', 0.6, 'Moderate - cloud can define ranges'),
    ('ranging', 'kama', 0.7, 'Good - adapts to ranging conditions'),
    ('ranging', 'momentum', 0.5, 'Poor - little momentum in ranges'),
    ('ranging', 'zero_lag', 0.6, 'Moderate - can catch range breaks'),
    ('ranging', 'bb_supertrend', 0.7, 'Good - trend component helps'),
    ('ranging', 'smc', 1.0, 'PERFECT - smart money excels in ranges'),
    ('ranging', 'smc_simple', 1.0, 'PERFECT - SMC excellent in ranges'),
    ('ranging', 'mean_reversion', 1.0, 'PERFECT - designed for ranges'),
    ('ranging', 'ranging_market', 1.0, 'PERFECT - specifically for ranges'),
    ('ranging', 'bollinger', 1.0, 'PERFECT - mean reversion in bands'),
    ('ranging', 'scalping', 0.9, 'Excellent - ranges perfect for scalping'),

    -- BREAKOUT REGIME
    ('breakout', 'ema', 0.95, 'EXCELLENT - EMA crossovers confirm breakouts'),
    ('breakout', 'smart_money_ema', 0.95, 'Excellent - EMA with smart money confirmation'),
    ('breakout', 'ema_double', 0.95, 'EXCELLENT - FVG confirms breakout momentum'),
    ('breakout', 'macd', 0.9, 'Very good - momentum confirmation'),
    ('breakout', 'smart_money_macd', 0.9, 'Very good - MACD with smart money'),
    ('breakout', 'ichimoku', 0.8, 'Good - cloud breakouts'),
    ('breakout', 'kama', 0.9, 'Excellent - adapts to breakout volatility'),
    ('breakout', 'momentum', 1.0, 'PERFECT - momentum drives breakouts'),
    ('breakout', 'zero_lag', 0.95, 'Excellent - fast breakout detection'),
    ('breakout', 'bb_supertrend', 1.0, 'PERFECT - designed for breakouts'),
    ('breakout', 'smc', 0.85, 'Very good - smart money breakout detection'),
    ('breakout', 'smc_simple', 0.85, 'Very good - SMC breakout detection'),
    ('breakout', 'bollinger', 0.9, 'Excellent - band breakouts'),
    ('breakout', 'mean_reversion', 0.2, 'Very poor - fights breakout momentum'),
    ('breakout', 'ranging_market', 0.15, 'Very poor - opposite of breakouts'),
    ('breakout', 'scalping', 0.6, 'Moderate - volatility can help'),

    -- CONSOLIDATION REGIME
    ('consolidation', 'ema', 0.7, 'GOOD - can define consolidation boundaries'),
    ('consolidation', 'smart_money_ema', 0.75, 'Good - EMA with smart money insight'),
    ('consolidation', 'ema_double', 0.5, 'POOR - ADX filters but some false signals possible'),
    ('consolidation', 'macd', 0.45, 'POOR - consolidation lacks momentum'),
    ('consolidation', 'smart_money_macd', 0.5, 'MODERATE - slightly better with smart money'),
    ('consolidation', 'ichimoku', 0.7, 'Good - cloud consolidation patterns'),
    ('consolidation', 'kama', 0.7, 'Good - adapts to low volatility'),
    ('consolidation', 'momentum', 0.4, 'Poor - little momentum in consolidation'),
    ('consolidation', 'zero_lag', 0.5, 'Poor - needs more movement'),
    ('consolidation', 'bb_supertrend', 0.6, 'Moderate - trend component struggles'),
    ('consolidation', 'smc', 1.0, 'PERFECT - smart money excels here'),
    ('consolidation', 'smc_simple', 0.95, 'Excellent - SMC works well'),
    ('consolidation', 'mean_reversion', 1.0, 'PERFECT - consolidation = mean reversion'),
    ('consolidation', 'ranging_market', 1.0, 'PERFECT - similar to ranging'),
    ('consolidation', 'bollinger', 0.8, 'Good - mean reversion in consolidation'),
    ('consolidation', 'scalping', 0.85, 'Very good - tight ranges for scalping'),

    -- HIGH VOLATILITY REGIME
    ('high_volatility', 'ema', 1.0, 'PERFECT - EMA handles volatility well'),
    ('high_volatility', 'smart_money_ema', 1.0, 'PERFECT - EMA with smart money'),
    ('high_volatility', 'ema_double', 0.9, 'EXCELLENT - confirmation filters help'),
    ('high_volatility', 'macd', 0.6, 'MODERATE - high vol can be choppy'),
    ('high_volatility', 'smart_money_macd', 0.65, 'MODERATE - slightly better with smart money'),
    ('high_volatility', 'ichimoku', 0.8, 'Good - cloud analysis works'),
    ('high_volatility', 'kama', 1.0, 'PERFECT - adapts to high volatility'),
    ('high_volatility', 'momentum', 1.0, 'PERFECT - volatility creates momentum'),
    ('high_volatility', 'zero_lag', 1.0, 'PERFECT - designed for fast markets'),
    ('high_volatility', 'bb_supertrend', 1.0, 'PERFECT - volatility breakouts'),
    ('high_volatility', 'smc', 0.85, 'Very good - smart money in volatility'),
    ('high_volatility', 'smc_simple', 0.85, 'Very good - SMC handles volatility'),
    ('high_volatility', 'bollinger', 0.8, 'Good - bands expand with volatility'),
    ('high_volatility', 'mean_reversion', 0.3, 'Poor - volatility fights mean reversion'),
    ('high_volatility', 'ranging_market', 0.2, 'Very poor - opposite of ranging'),
    ('high_volatility', 'scalping', 0.9, 'Excellent - volatility creates opportunities'),

    -- LOW VOLATILITY REGIME
    ('low_volatility', 'ema', 1.0, 'PERFECT - smooth trends in low vol'),
    ('low_volatility', 'smart_money_ema', 1.0, 'PERFECT - EMA with smart money'),
    ('low_volatility', 'ema_double', 0.85, 'VERY GOOD - works in smooth trends'),
    ('low_volatility', 'macd', 0.85, 'Good - works with tighter thresholds'),
    ('low_volatility', 'smart_money_macd', 0.85, 'Good - MACD with smart money'),
    ('low_volatility', 'ichimoku', 0.8, 'Good - cloud analysis still works'),
    ('low_volatility', 'kama', 0.8, 'Good - adapts to low volatility'),
    ('low_volatility', 'momentum', 0.4, 'Poor - little momentum in low vol'),
    ('low_volatility', 'zero_lag', 0.5, 'Poor - needs more movement'),
    ('low_volatility', 'bb_supertrend', 0.6, 'Moderate - trend component struggles'),
    ('low_volatility', 'smc', 1.0, 'PERFECT - smart money in quiet markets'),
    ('low_volatility', 'smc_simple', 1.0, 'PERFECT - SMC excellent in quiet'),
    ('low_volatility', 'bollinger', 1.0, 'PERFECT - tight bands in low vol'),
    ('low_volatility', 'mean_reversion', 1.0, 'PERFECT - low vol enables mean reversion'),
    ('low_volatility', 'ranging_market', 1.0, 'PERFECT - low vol creates ranges'),
    ('low_volatility', 'scalping', 0.8, 'Good - tight spreads in low vol'),

    -- MEDIUM VOLATILITY REGIME (Goldilocks zone)
    ('medium_volatility', 'ema', 1.0, 'PERFECT - ideal conditions for EMA'),
    ('medium_volatility', 'smart_money_ema', 1.0, 'PERFECT - EMA with smart money'),
    ('medium_volatility', 'ema_double', 1.0, 'PERFECT - ideal for confirmed crossovers'),
    ('medium_volatility', 'macd', 1.0, 'PERFECT - ideal MACD conditions'),
    ('medium_volatility', 'smart_money_macd', 1.0, 'PERFECT - MACD with smart money'),
    ('medium_volatility', 'ichimoku', 1.0, 'PERFECT - ideal cloud conditions'),
    ('medium_volatility', 'kama', 1.0, 'PERFECT - balanced adaptive conditions'),
    ('medium_volatility', 'momentum', 0.9, 'Excellent - sufficient momentum'),
    ('medium_volatility', 'zero_lag', 0.95, 'Excellent - good response time'),
    ('medium_volatility', 'bb_supertrend', 0.9, 'Excellent - balanced trend/volatility'),
    ('medium_volatility', 'smc', 0.9, 'Excellent - smart money analysis'),
    ('medium_volatility', 'smc_simple', 0.95, 'Excellent - SMC works great'),
    ('medium_volatility', 'bollinger', 0.85, 'Very good - moderate band expansion'),
    ('medium_volatility', 'mean_reversion', 0.8, 'Good - still some mean reversion'),
    ('medium_volatility', 'ranging_market', 0.7, 'Good - some ranging behavior'),
    ('medium_volatility', 'scalping', 0.75, 'Good - balanced conditions'),

    -- UNKNOWN REGIME (conservative defaults)
    ('unknown', 'ema', 0.8, 'GOOD - safe default for foundational strategy'),
    ('unknown', 'smart_money_ema', 0.8, 'Good - EMA with smart money'),
    ('unknown', 'ema_double', 0.75, 'GOOD - multiple confirmations help'),
    ('unknown', 'macd', 0.7, 'Good - proven versatile strategy'),
    ('unknown', 'smart_money_macd', 0.7, 'Good - MACD with smart money'),
    ('unknown', 'ichimoku', 0.7, 'Good - comprehensive analysis'),
    ('unknown', 'kama', 0.7, 'Good - adaptive nature helps'),
    ('unknown', 'momentum', 0.6, 'Moderate - momentum can be risky'),
    ('unknown', 'zero_lag', 0.6, 'Moderate - fast response can be risky'),
    ('unknown', 'bb_supertrend', 0.6, 'Moderate - trend component helps'),
    ('unknown', 'smc', 0.7, 'Good - smart money provides insight'),
    ('unknown', 'smc_simple', 0.7, 'Good - SMC provides insight'),
    ('unknown', 'bollinger', 0.6, 'Moderate - bands provide guidance'),
    ('unknown', 'mean_reversion', 0.6, 'Moderate - safer in uncertainty'),
    ('unknown', 'ranging_market', 0.5, 'Moderate - specific strategy'),
    ('unknown', 'scalping', 0.4, 'Poor - uncertainty bad for scalping')

ON CONFLICT (regime, strategy) DO UPDATE SET
    confidence_modifier = EXCLUDED.confidence_modifier,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE intelligence_global_config IS 'Global configuration parameters for Market Intelligence system';
COMMENT ON TABLE intelligence_presets IS 'Intelligence preset definitions with thresholds';
COMMENT ON TABLE intelligence_preset_components IS 'Component enablement per intelligence preset';
COMMENT ON TABLE intelligence_regime_modifiers IS 'Regime-strategy confidence modifiers for probabilistic filtering';
COMMENT ON TABLE intelligence_config_audit IS 'Audit trail for intelligence configuration changes';

-- ============================================================================
-- VERIFY MIGRATION
-- ============================================================================
DO $$
DECLARE
    v_global_count INTEGER;
    v_preset_count INTEGER;
    v_component_count INTEGER;
    v_modifier_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_global_count FROM intelligence_global_config;
    SELECT COUNT(*) INTO v_preset_count FROM intelligence_presets;
    SELECT COUNT(*) INTO v_component_count FROM intelligence_preset_components;
    SELECT COUNT(*) INTO v_modifier_count FROM intelligence_regime_modifiers;

    RAISE NOTICE '✅ Intelligence Config Migration Complete:';
    RAISE NOTICE '   - Global config parameters: %', v_global_count;
    RAISE NOTICE '   - Presets defined: %', v_preset_count;
    RAISE NOTICE '   - Preset components: %', v_component_count;
    RAISE NOTICE '   - Regime modifiers: %', v_modifier_count;
END $$;
