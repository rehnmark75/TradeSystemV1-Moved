-- Migration: 013_add_premarket_entry_price.sql
-- Date: 2024-12-29
-- Description: Add premarket entry price column to stock_scanner_signals
--              to store updated entry prices from pre-market session

-- Add premarket entry price column
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS premarket_entry_price DECIMAL(12, 4);

-- Add premarket gap percentage
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS premarket_gap_percent DECIMAL(6, 2);

-- Add premarket update timestamp
ALTER TABLE stock_scanner_signals
ADD COLUMN IF NOT EXISTS premarket_updated_at TIMESTAMPTZ;

-- Add index for signals with premarket data
CREATE INDEX IF NOT EXISTS idx_scanner_signals_premarket
    ON stock_scanner_signals(premarket_updated_at DESC)
    WHERE premarket_entry_price IS NOT NULL;

-- Comments
COMMENT ON COLUMN stock_scanner_signals.premarket_entry_price IS
    'Entry price updated from pre-market quote (9:00 AM ET). Original entry_price preserved.';

COMMENT ON COLUMN stock_scanner_signals.premarket_gap_percent IS
    'Gap percentage from previous close to premarket price';

COMMENT ON COLUMN stock_scanner_signals.premarket_updated_at IS
    'Timestamp when premarket data was last updated';
