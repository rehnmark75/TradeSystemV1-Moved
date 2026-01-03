-- ============================================================================
-- TEMPLATE Strategy Database Configuration
-- ============================================================================
--
-- INSTRUCTIONS:
-- 1. Copy this file to migrations/create_template_strategy_config.sql
-- 2. Replace all instances of 'template' with your strategy name (lowercase)
-- 3. Replace all instances of 'TEMPLATE' with your strategy name (uppercase)
-- 4. Customize parameters for your strategy's needs
-- 5. Run: docker exec postgres psql -U postgres -d strategy_config -f /path/to/migration.sql
--
-- TABLES CREATED:
-- - template_global_config: Global strategy parameters
-- - template_pair_overrides: Per-pair parameter overrides
-- - template_config_audit: Change history (optional)
-- ============================================================================

-- Connect to strategy_config database
-- \c strategy_config

-- ============================================================================
-- 1. GLOBAL CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS template_global_config (
    id SERIAL PRIMARY KEY,

    -- Parameter identification
    parameter_name VARCHAR(100) NOT NULL UNIQUE,
    parameter_value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',  -- string, int, float, bool, json

    -- Grouping and display
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    display_order INTEGER DEFAULT 0,
    description TEXT,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_editable BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 2. INSERT DEFAULT PARAMETERS
-- ============================================================================

INSERT INTO template_global_config
    (parameter_name, parameter_value, value_type, category, description, display_order)
VALUES
    -- Strategy identification
    ('strategy_name', 'TEMPLATE', 'string', 'general', 'Strategy name', 1),
    ('version', '1.0.0', 'string', 'general', 'Strategy version', 2),
    ('enabled', 'true', 'bool', 'general', 'Enable/disable strategy', 3),

    -- Timeframes
    ('htf_timeframe', '4h', 'string', 'timeframes', 'Higher timeframe for bias', 10),
    ('trigger_timeframe', '1h', 'string', 'timeframes', 'Trigger timeframe', 11),
    ('entry_timeframe', '15m', 'string', 'timeframes', 'Entry timeframe', 12),

    -- Core parameters (TODO: customize for your strategy)
    ('ema_period', '50', 'int', 'indicators', 'EMA period for trend detection', 20),
    ('rsi_period', '14', 'int', 'indicators', 'RSI period', 21),
    ('atr_period', '14', 'int', 'indicators', 'ATR period', 22),

    -- Confidence thresholds
    ('min_confidence', '0.60', 'float', 'confidence', 'Minimum confidence to take trade', 30),
    ('max_confidence', '0.90', 'float', 'confidence', 'Maximum confidence cap', 31),

    -- Stop Loss / Take Profit defaults
    ('fixed_stop_loss_pips', '15.0', 'float', 'risk', 'Default stop loss in pips', 40),
    ('fixed_take_profit_pips', '25.0', 'float', 'risk', 'Default take profit in pips', 41),
    ('sl_buffer_pips', '3.0', 'float', 'risk', 'Stop loss buffer in pips', 42),
    ('sl_atr_multiplier', '1.0', 'float', 'risk', 'ATR multiplier for SL calculation', 43),
    ('min_risk_reward', '1.5', 'float', 'risk', 'Minimum risk:reward ratio', 44),

    -- Filters
    ('volume_filter_enabled', 'true', 'bool', 'filters', 'Enable volume confirmation', 50),
    ('volume_threshold', '1.5', 'float', 'filters', 'Volume spike threshold (x average)', 51),
    ('atr_filter_enabled', 'true', 'bool', 'filters', 'Enable ATR volatility filter', 52),
    ('min_atr_pips', '5.0', 'float', 'filters', 'Minimum ATR in pips', 53),

    -- Cooldown
    ('signal_cooldown_minutes', '60', 'int', 'cooldown', 'Cooldown between signals per pair', 60),

    -- Enabled pairs (empty = all pairs)
    ('enabled_pairs', '[]', 'json', 'pairs', 'List of enabled epics (empty = all)', 70)

ON CONFLICT (parameter_name) DO NOTHING;


-- ============================================================================
-- 3. PER-PAIR OVERRIDES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS template_pair_overrides (
    id SERIAL PRIMARY KEY,

    -- Pair identification
    epic VARCHAR(50) NOT NULL UNIQUE,
    pair_name VARCHAR(10),

    -- Override parameters (NULL means use global default)
    fixed_stop_loss_pips FLOAT,
    fixed_take_profit_pips FLOAT,
    sl_buffer_pips FLOAT,
    min_confidence FLOAT,
    max_confidence FLOAT,
    signal_cooldown_minutes INTEGER,

    -- Pair-specific flags
    is_enabled BOOLEAN DEFAULT TRUE,
    is_traded BOOLEAN DEFAULT TRUE,

    -- Notes
    notes TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert common forex pairs with defaults (customize as needed)
INSERT INTO template_pair_overrides (epic, pair_name, is_enabled)
VALUES
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', TRUE),
    ('CS.D.GBPUSD.MINI.IP', 'GBPUSD', TRUE),
    ('CS.D.USDJPY.MINI.IP', 'USDJPY', TRUE),
    ('CS.D.AUDUSD.MINI.IP', 'AUDUSD', TRUE),
    ('CS.D.USDCAD.MINI.IP', 'USDCAD', TRUE),
    ('CS.D.USDCHF.MINI.IP', 'USDCHF', TRUE),
    ('CS.D.NZDUSD.MINI.IP', 'NZDUSD', TRUE),
    ('CS.D.EURJPY.MINI.IP', 'EURJPY', TRUE),
    ('CS.D.GBPJPY.MINI.IP', 'GBPJPY', TRUE),
    ('CS.D.AUDJPY.MINI.IP', 'AUDJPY', TRUE)
ON CONFLICT (epic) DO NOTHING;


-- ============================================================================
-- 4. AUDIT TABLE (Optional - for tracking configuration changes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS template_config_audit (
    id SERIAL PRIMARY KEY,

    -- Change details
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    parameter_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    change_type VARCHAR(20) NOT NULL,  -- INSERT, UPDATE, DELETE

    -- Context
    changed_by VARCHAR(100) DEFAULT 'system',
    change_reason TEXT,

    -- Timestamp
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- ============================================================================
-- 5. HELPER FUNCTIONS
-- ============================================================================

-- Function to get parameter value with type conversion
CREATE OR REPLACE FUNCTION get_template_config(param_name VARCHAR)
RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    SELECT parameter_value INTO result
    FROM template_global_config
    WHERE parameter_name = param_name AND is_active = TRUE;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to get pair-specific SL (with fallback to global)
CREATE OR REPLACE FUNCTION get_template_pair_sl(p_epic VARCHAR)
RETURNS FLOAT AS $$
DECLARE
    pair_sl FLOAT;
    global_sl FLOAT;
BEGIN
    -- Try pair-specific first
    SELECT fixed_stop_loss_pips INTO pair_sl
    FROM template_pair_overrides
    WHERE epic = p_epic AND is_enabled = TRUE;

    IF pair_sl IS NOT NULL THEN
        RETURN pair_sl;
    END IF;

    -- Fallback to global
    SELECT parameter_value::FLOAT INTO global_sl
    FROM template_global_config
    WHERE parameter_name = 'fixed_stop_loss_pips' AND is_active = TRUE;

    RETURN COALESCE(global_sl, 15.0);
END;
$$ LANGUAGE plpgsql;

-- Function to get pair-specific TP (with fallback to global)
CREATE OR REPLACE FUNCTION get_template_pair_tp(p_epic VARCHAR)
RETURNS FLOAT AS $$
DECLARE
    pair_tp FLOAT;
    global_tp FLOAT;
BEGIN
    -- Try pair-specific first
    SELECT fixed_take_profit_pips INTO pair_tp
    FROM template_pair_overrides
    WHERE epic = p_epic AND is_enabled = TRUE;

    IF pair_tp IS NOT NULL THEN
        RETURN pair_tp;
    END IF;

    -- Fallback to global
    SELECT parameter_value::FLOAT INTO global_tp
    FROM template_global_config
    WHERE parameter_name = 'fixed_take_profit_pips' AND is_active = TRUE;

    RETURN COALESCE(global_tp, 25.0);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- 6. UPDATE TRIGGER (for audit tracking)
-- ============================================================================

CREATE OR REPLACE FUNCTION template_config_audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        INSERT INTO template_config_audit
            (table_name, record_id, parameter_name, old_value, new_value, change_type)
        VALUES
            (TG_TABLE_NAME, OLD.id, OLD.parameter_name, OLD.parameter_value, NEW.parameter_value, 'UPDATE');
        NEW.updated_at = NOW();
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO template_config_audit
            (table_name, record_id, parameter_name, old_value, new_value, change_type)
        VALUES
            (TG_TABLE_NAME, NEW.id, NEW.parameter_name, NULL, NEW.parameter_value, 'INSERT');
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO template_config_audit
            (table_name, record_id, parameter_name, old_value, new_value, change_type)
        VALUES
            (TG_TABLE_NAME, OLD.id, OLD.parameter_name, OLD.parameter_value, NULL, 'DELETE');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (uncomment if audit tracking desired)
-- DROP TRIGGER IF EXISTS template_global_config_audit ON template_global_config;
-- CREATE TRIGGER template_global_config_audit
--     AFTER INSERT OR UPDATE OR DELETE ON template_global_config
--     FOR EACH ROW EXECUTE FUNCTION template_config_audit_trigger();


-- ============================================================================
-- 7. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_template_global_config_category
    ON template_global_config(category);

CREATE INDEX IF NOT EXISTS idx_template_global_config_active
    ON template_global_config(is_active);

CREATE INDEX IF NOT EXISTS idx_template_pair_overrides_enabled
    ON template_pair_overrides(is_enabled);


-- ============================================================================
-- 8. VERIFICATION QUERIES
-- ============================================================================

-- View all global parameters
-- SELECT parameter_name, parameter_value, value_type, category
-- FROM template_global_config
-- WHERE is_active = TRUE
-- ORDER BY category, display_order;

-- View pair overrides
-- SELECT epic, pair_name, fixed_stop_loss_pips, fixed_take_profit_pips
-- FROM template_pair_overrides
-- WHERE is_enabled = TRUE
-- ORDER BY pair_name;

-- Test helper functions
-- SELECT get_template_config('min_confidence');
-- SELECT get_template_pair_sl('CS.D.EURUSD.CEEM.IP');


-- ============================================================================
-- DONE
-- ============================================================================

SELECT 'TEMPLATE strategy configuration tables created successfully' AS status;
