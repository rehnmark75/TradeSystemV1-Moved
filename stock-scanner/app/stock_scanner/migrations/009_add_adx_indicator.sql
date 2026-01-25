-- Migration 009: Add ADX indicator to stock_screening_metrics
-- ADX (Average Directional Index) is used to measure trend strength
-- Required for optimized EMA Pullback strategy (ADX > 20 filter)

-- Add ADX column to stock_screening_metrics
ALTER TABLE stock_screening_metrics
ADD COLUMN IF NOT EXISTS adx DECIMAL(6, 2);

-- Add index for filtering by ADX
CREATE INDEX IF NOT EXISTS idx_stock_screening_metrics_adx
ON stock_screening_metrics (adx)
WHERE adx IS NOT NULL;

-- Comment
COMMENT ON COLUMN stock_screening_metrics.adx IS 'Average Directional Index (14-period) - trend strength indicator';
