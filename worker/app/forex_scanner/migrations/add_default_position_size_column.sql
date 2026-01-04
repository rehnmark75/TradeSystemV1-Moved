-- Migration: Add default_position_size column to scanner_global_config
-- Date: 2026-01-04
-- Purpose: Support OrderManager database-only configuration
-- Applied via: docker exec postgres psql -U postgres -d strategy_config -f /path/to/this/file.sql

-- Add default_position_size column if it doesn't exist
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS default_position_size NUMERIC(5,2) DEFAULT 1.0;

-- Add comment for documentation
COMMENT ON COLUMN scanner_global_config.default_position_size IS 'Default position size for order execution (1.0 = 1 mini lot)';

-- Update any NULL values to default
UPDATE scanner_global_config
SET default_position_size = 1.0
WHERE default_position_size IS NULL;

-- Verify the column was added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'scanner_global_config'
AND column_name = 'default_position_size';
