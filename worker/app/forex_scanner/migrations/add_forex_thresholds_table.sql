-- Migration: Add forex_thresholds configuration table
-- Purpose: Store MACD thresholds in database to prevent weak signals from passing through
-- Date: 2025-08-26

-- Create the forex_thresholds table
CREATE TABLE IF NOT EXISTS forex_thresholds (
    epic VARCHAR(50) PRIMARY KEY,
    base_threshold DECIMAL(10,8) NOT NULL,
    strength_thresholds JSONB NOT NULL DEFAULT '{}',
    session_multipliers JSONB DEFAULT '{"london": 1.0, "new_york": 1.1, "asian": 0.8}',
    pair_type VARCHAR(50),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    notes TEXT
);

-- Add index for active records
CREATE INDEX IF NOT EXISTS idx_forex_thresholds_active ON forex_thresholds(is_active);

-- Insert corrected thresholds for major pairs
INSERT INTO forex_thresholds (epic, base_threshold, strength_thresholds, pair_type, notes)
VALUES 
    -- Major USD pairs (CORRECTED - 5-6x increase from old values)
    ('CS.D.EURUSD.CEEM.IP', 0.00005000, 
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'eur_major', 'EUR/USD - Corrected from 0.000008'),
    
    ('CS.D.GBPUSD.MINI.IP', 0.00008000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'gbp_volatile', 'GBP/USD - Corrected from 0.000015'),
    
    ('CS.D.AUDUSD.MINI.IP', 0.00006000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'aud_commodity', 'AUD/USD - Corrected from 0.000012'),
    
    ('CS.D.NZDUSD.MINI.IP', 0.00009000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'nzd_commodity', 'NZD/USD - Corrected from 0.000018'),
    
    ('CS.D.USDCAD.MINI.IP', 0.00005000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'usdcad_oil', 'USD/CAD - Corrected from 0.000010'),
    
    ('CS.D.USDCHF.MINI.IP', 0.00008000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'usdchf_safe_haven', 'USD/CHF - Corrected from 0.000015'),
    
    -- JPY pairs (CORRECTED - 10x increase from old values)
    ('CS.D.USDJPY.MINI.IP', 0.00800000,
     '{"moderate": 0.005, "strong": 0.010, "very_strong": 0.020}',
     'usdjpy_stable', 'USD/JPY - Corrected from 0.0008'),
    
    ('CS.D.EURJPY.MINI.IP', 0.01000000,
     '{"moderate": 0.005, "strong": 0.010, "very_strong": 0.020}',
     'eurjpy_cross', 'EUR/JPY - Corrected from 0.0012'),
    
    ('CS.D.GBPJPY.MINI.IP', 0.01200000,
     '{"moderate": 0.005, "strong": 0.010, "very_strong": 0.020}',
     'gbpjpy_volatile', 'GBP/JPY - Corrected from 0.0015'),
    
    ('CS.D.AUDJPY.MINI.IP', 0.00800000,
     '{"moderate": 0.005, "strong": 0.010, "very_strong": 0.020}',
     'audjpy_carry', 'AUD/JPY - Corrected from 0.0010'),
    
    -- Cross pairs
    ('CS.D.EURGBP.MINI.IP', 0.00006000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'eur_cross', 'EUR/GBP - Corrected from 0.000010'),
    
    ('CS.D.EURAUD.MINI.IP', 0.00007000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'eur_cross', 'EUR/AUD - Corrected from 0.000014'),
    
    ('CS.D.GBPAUD.MINI.IP', 0.00008000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'gbp_cross', 'GBP/AUD - Corrected from 0.000016'),
    
    ('CS.D.AUDCAD.MINI.IP', 0.00006000,
     '{"moderate": 0.0004, "strong": 0.0008, "very_strong": 0.0012}',
     'aud_cross', 'AUD/CAD - Corrected from 0.000013')
ON CONFLICT (epic) DO UPDATE SET
    base_threshold = EXCLUDED.base_threshold,
    strength_thresholds = EXCLUDED.strength_thresholds,
    pair_type = EXCLUDED.pair_type,
    last_updated = CURRENT_TIMESTAMP,
    notes = EXCLUDED.notes;

-- Create function to get threshold with fallback
CREATE OR REPLACE FUNCTION get_macd_threshold(p_epic VARCHAR)
RETURNS DECIMAL AS $$
DECLARE
    v_threshold DECIMAL;
BEGIN
    -- Try to get from database
    SELECT base_threshold INTO v_threshold
    FROM forex_thresholds
    WHERE epic = p_epic AND is_active = true;
    
    -- If not found, use corrected fallback based on pair type
    IF v_threshold IS NULL THEN
        IF p_epic LIKE '%JPY%' THEN
            v_threshold := 0.008;  -- Corrected JPY fallback
        ELSE
            v_threshold := 0.00008;  -- Corrected non-JPY fallback
        END IF;
    END IF;
    
    RETURN v_threshold;
END;
$$ LANGUAGE plpgsql;

-- Add comment explaining the critical nature of these thresholds
COMMENT ON TABLE forex_thresholds IS 'Critical MACD thresholds for forex pairs. These values prevent weak signals from triggering trades. Updated 2025-08-26 to fix issue where 0.00002 signals were passing through.';