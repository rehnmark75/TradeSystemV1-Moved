-- Migration: Convert ARRAY columns to JSONB for better compatibility
-- This fixes the psycopg2 "List argument must consist only of dictionaries" error

ALTER TABLE backtest_signals
    ALTER COLUMN validation_flags TYPE JSONB USING validation_flags::text::jsonb,
    ALTER COLUMN validation_reasons TYPE JSONB USING validation_reasons::text::jsonb;

-- Verify the change
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'backtest_signals'
AND column_name IN ('validation_flags', 'validation_reasons');
