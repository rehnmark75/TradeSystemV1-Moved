-- ============================================================================
-- FVG Retest Strategy Database Configuration
-- ============================================================================
--
-- Dual-mode strategy: Type A (Deep Value / FVG Tap) + Type B (Institutional Initiation)
-- 1H macro → 5m BOS classification → 5m entry (immediate or deferred)
--
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/create_fvg_retest_config.sql
-- ============================================================================

-- ============================================================================
-- 1. GLOBAL CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fvg_retest_global_config (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(100) NOT NULL UNIQUE,
    parameter_value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    display_order INTEGER DEFAULT 0,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_editable BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 2. INSERT DEFAULT PARAMETERS
-- ============================================================================

INSERT INTO fvg_retest_global_config
    (parameter_name, parameter_value, value_type, category, description, display_order)
VALUES
    -- Strategy identification
    ('strategy_name', 'FVG_RETEST', 'string', 'general', 'Strategy name', 1),
    ('version', '1.0.0', 'string', 'general', 'Strategy version', 2),
    ('enabled', 'true', 'bool', 'general', 'Enable/disable strategy', 3),

    -- Timeframes
    ('htf_timeframe', '1h', 'string', 'timeframes', 'Higher timeframe for 200 EMA bias', 10),
    ('trigger_timeframe', '5m', 'string', 'timeframes', 'BOS detection and entry timeframe', 11),

    -- HTF filter
    ('htf_ema_period', '200', 'int', 'htf', '1H EMA period for trend filter', 20),

    -- Shared risk parameters
    ('fixed_stop_loss_pips', '12.0', 'float', 'risk', 'Default stop loss in pips', 30),
    ('fixed_take_profit_pips', '20.0', 'float', 'risk', 'Default take profit in pips', 31),
    ('sl_buffer_pips', '3.0', 'float', 'risk', 'Stop loss buffer beyond level', 32),
    ('min_rr_ratio', '1.5', 'float', 'risk', 'Minimum risk:reward ratio', 33),

    -- Confidence thresholds
    ('min_confidence', '0.50', 'float', 'confidence', 'Minimum confidence to take trade', 40),
    ('max_confidence', '0.90', 'float', 'confidence', 'Maximum confidence cap', 41),

    -- Swing detection
    ('swing_lookback_bars', '20', 'int', 'swing', 'Bars to look back for swing points', 50),
    ('swing_strength_bars', '2', 'int', 'swing', 'Bars on each side to confirm swing', 51),
    ('atr_period', '14', 'int', 'swing', 'ATR period for volatility measurement', 52),

    -- Type A: FVG Tap (Deep Value) parameters
    ('fvg_min_size_pips', '3.0', 'float', 'type_a', 'Minimum FVG size in pips', 60),
    ('fvg_max_age_bars', '20', 'int', 'type_a', 'Max bars before FVG expires', 61),
    ('fvg_max_fill_pct', '0.80', 'float', 'type_a', 'Max fill percentage before FVG expires', 62),
    ('setup_expiry_hours', '4.0', 'float', 'type_a', 'Hours before pending setup expires', 63),
    ('max_pending_per_pair', '3', 'int', 'type_a', 'Max pending setups per pair', 64),

    -- Type B: Institutional Initiation parameters
    ('initiation_enabled', 'true', 'bool', 'type_b', 'Enable Type B immediate entries', 70),
    ('displacement_atr_multiplier', '1.5', 'float', 'type_b', 'Break candle body must exceed ATR * this', 71),
    ('follow_through_candles', '2', 'int', 'type_b', 'Candles closing in BOS direction after break', 72),
    ('volume_threshold_multiplier', '1.0', 'float', 'type_b', 'Break volume must exceed SMA * this', 73),

    -- Volume
    ('volume_sma_period', '20', 'int', 'volume', 'Volume SMA period for comparison', 80),

    -- Cooldown
    ('signal_cooldown_minutes', '60', 'int', 'cooldown', 'Cooldown between signals per pair', 90),

    -- Enabled pairs (empty = all pairs from pair_overrides)
    ('enabled_pairs', '[]', 'json', 'pairs', 'List of enabled epics (empty = all)', 100)

ON CONFLICT (parameter_name) DO NOTHING;


-- ============================================================================
-- 3. PER-PAIR OVERRIDES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fvg_retest_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    pair_name VARCHAR(10),
    fixed_stop_loss_pips FLOAT,
    fixed_take_profit_pips FLOAT,
    sl_buffer_pips FLOAT,
    min_confidence FLOAT,
    max_confidence FLOAT,
    signal_cooldown_minutes INTEGER,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_traded BOOLEAN DEFAULT TRUE,
    parameter_overrides JSONB DEFAULT '{}',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert all 9 active pairs + GBPJPY
INSERT INTO fvg_retest_pair_overrides (epic, pair_name, is_enabled)
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
-- 4. AUDIT TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fvg_retest_config_audit (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    parameter_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    change_type VARCHAR(20) NOT NULL,
    changed_by VARCHAR(100) DEFAULT 'system',
    change_reason TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- ============================================================================
-- 5. HELPER FUNCTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION get_fvg_retest_config(param_name VARCHAR)
RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    SELECT parameter_value INTO result
    FROM fvg_retest_global_config
    WHERE parameter_name = param_name AND is_active = TRUE;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_fvg_retest_pair_sl(p_epic VARCHAR)
RETURNS FLOAT AS $$
DECLARE
    pair_sl FLOAT;
    global_sl FLOAT;
BEGIN
    SELECT fixed_stop_loss_pips INTO pair_sl
    FROM fvg_retest_pair_overrides
    WHERE epic = p_epic AND is_enabled = TRUE;

    IF pair_sl IS NOT NULL THEN
        RETURN pair_sl;
    END IF;

    SELECT parameter_value::FLOAT INTO global_sl
    FROM fvg_retest_global_config
    WHERE parameter_name = 'fixed_stop_loss_pips' AND is_active = TRUE;

    RETURN COALESCE(global_sl, 12.0);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_fvg_retest_pair_tp(p_epic VARCHAR)
RETURNS FLOAT AS $$
DECLARE
    pair_tp FLOAT;
    global_tp FLOAT;
BEGIN
    SELECT fixed_take_profit_pips INTO pair_tp
    FROM fvg_retest_pair_overrides
    WHERE epic = p_epic AND is_enabled = TRUE;

    IF pair_tp IS NOT NULL THEN
        RETURN pair_tp;
    END IF;

    SELECT parameter_value::FLOAT INTO global_tp
    FROM fvg_retest_global_config
    WHERE parameter_name = 'fixed_take_profit_pips' AND is_active = TRUE;

    RETURN COALESCE(global_tp, 20.0);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- 6. UPDATE TRIGGER
-- ============================================================================

CREATE OR REPLACE FUNCTION fvg_retest_config_audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        INSERT INTO fvg_retest_config_audit
            (table_name, record_id, parameter_name, old_value, new_value, change_type)
        VALUES
            (TG_TABLE_NAME, OLD.id, OLD.parameter_name, OLD.parameter_value, NEW.parameter_value, 'UPDATE');
        NEW.updated_at = NOW();
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO fvg_retest_config_audit
            (table_name, record_id, parameter_name, old_value, new_value, change_type)
        VALUES
            (TG_TABLE_NAME, NEW.id, NEW.parameter_name, NULL, NEW.parameter_value, 'INSERT');
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO fvg_retest_config_audit
            (table_name, record_id, parameter_name, old_value, new_value, change_type)
        VALUES
            (TG_TABLE_NAME, OLD.id, OLD.parameter_name, OLD.parameter_value, NULL, 'DELETE');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- 7. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_fvg_retest_global_config_category
    ON fvg_retest_global_config(category);

CREATE INDEX IF NOT EXISTS idx_fvg_retest_global_config_active
    ON fvg_retest_global_config(is_active);

CREATE INDEX IF NOT EXISTS idx_fvg_retest_pair_overrides_enabled
    ON fvg_retest_pair_overrides(is_enabled);


-- ============================================================================
-- DONE
-- ============================================================================

SELECT 'FVG_RETEST strategy configuration tables created successfully' AS status;
