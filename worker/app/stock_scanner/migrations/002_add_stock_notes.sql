-- Stock notes / journaling
CREATE TABLE IF NOT EXISTS stock_notes (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    note_text TEXT NOT NULL,
    context VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stock_notes_ticker_created
    ON stock_notes (ticker, created_at DESC);

CREATE OR REPLACE FUNCTION update_stock_notes_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_stock_notes_timestamp ON stock_notes;
CREATE TRIGGER update_stock_notes_timestamp
    BEFORE UPDATE ON stock_notes
    FOR EACH ROW
    EXECUTE FUNCTION update_stock_notes_timestamp();
