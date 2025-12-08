-- =============================================================================
-- STOCK SCANNER DATABASE MIGRATION
-- Migration: 003_add_fundamentals_columns.sql
-- Description: Add fundamental data columns to stock_instruments table
-- Database: stocks
-- =============================================================================

-- =============================================================================
-- ADD FUNDAMENTAL COLUMNS TO STOCK_INSTRUMENTS
-- These columns store data fetched from yfinance
-- =============================================================================

-- Earnings & Calendar Data
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS earnings_date DATE;
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS earnings_date_estimated BOOLEAN DEFAULT TRUE;
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS ex_dividend_date DATE;

-- Risk Metrics
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS beta DECIMAL(6,3);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS short_ratio DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS short_percent_float DECIMAL(6,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS shares_short BIGINT;

-- Ownership
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS institutional_percent DECIMAL(6,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS insider_percent DECIMAL(6,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS shares_outstanding BIGINT;
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS shares_float BIGINT;

-- Valuation Metrics
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS forward_pe DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS trailing_pe DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS price_to_book DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS price_to_sales DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS peg_ratio DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS enterprise_to_ebitda DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS enterprise_to_revenue DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS enterprise_value BIGINT;

-- Growth Metrics
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS revenue_growth DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS earnings_growth DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS earnings_quarterly_growth DECIMAL(8,2);

-- Profitability Metrics
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS profit_margin DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS operating_margin DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS gross_margin DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS return_on_equity DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS return_on_assets DECIMAL(8,2);

-- Financial Health
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS debt_to_equity DECIMAL(10,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS current_ratio DECIMAL(8,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS quick_ratio DECIMAL(8,2);

-- Dividend Info
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS dividend_yield DECIMAL(6,2);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS dividend_rate DECIMAL(10,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS payout_ratio DECIMAL(8,2);

-- 52-Week Data
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS fifty_two_week_high DECIMAL(12,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS fifty_two_week_low DECIMAL(12,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS fifty_two_week_change DECIMAL(8,2);

-- Moving Averages (from yfinance, for quick reference)
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS fifty_day_average DECIMAL(12,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS two_hundred_day_average DECIMAL(12,4);

-- Analyst Data
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS analyst_rating VARCHAR(30);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS target_price DECIMAL(12,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS target_high DECIMAL(12,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS target_low DECIMAL(12,4);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS number_of_analysts INTEGER;

-- Company Description
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS business_summary TEXT;
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS website VARCHAR(255);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS country VARCHAR(50);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS city VARCHAR(100);
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS employee_count INTEGER;

-- Sync Tracking
ALTER TABLE stock_instruments ADD COLUMN IF NOT EXISTS fundamentals_updated_at TIMESTAMP;

-- =============================================================================
-- INDEXES FOR FUNDAMENTAL QUERIES
-- =============================================================================

-- High short interest screening
CREATE INDEX IF NOT EXISTS idx_stock_instruments_short_interest
    ON stock_instruments(short_percent_float DESC)
    WHERE short_percent_float IS NOT NULL AND is_active = TRUE;

-- Upcoming earnings filter
CREATE INDEX IF NOT EXISTS idx_stock_instruments_earnings
    ON stock_instruments(earnings_date)
    WHERE earnings_date IS NOT NULL AND is_active = TRUE;

-- Value screening (low P/E)
CREATE INDEX IF NOT EXISTS idx_stock_instruments_pe
    ON stock_instruments(trailing_pe)
    WHERE trailing_pe IS NOT NULL AND trailing_pe > 0 AND is_active = TRUE;

-- Growth screening
CREATE INDEX IF NOT EXISTS idx_stock_instruments_growth
    ON stock_instruments(earnings_growth DESC)
    WHERE earnings_growth IS NOT NULL AND is_active = TRUE;

-- Profitability screening
CREATE INDEX IF NOT EXISTS idx_stock_instruments_profitability
    ON stock_instruments(return_on_equity DESC)
    WHERE return_on_equity IS NOT NULL AND is_active = TRUE;

-- Financial health screening
CREATE INDEX IF NOT EXISTS idx_stock_instruments_debt
    ON stock_instruments(debt_to_equity)
    WHERE debt_to_equity IS NOT NULL AND is_active = TRUE;

-- Dividend screening
CREATE INDEX IF NOT EXISTS idx_stock_instruments_dividend
    ON stock_instruments(dividend_yield DESC)
    WHERE dividend_yield IS NOT NULL AND dividend_yield > 0 AND is_active = TRUE;

-- 52-week position screening
CREATE INDEX IF NOT EXISTS idx_stock_instruments_52w
    ON stock_instruments(fifty_two_week_high, fifty_two_week_low)
    WHERE is_active = TRUE;

-- Fundamentals freshness
CREATE INDEX IF NOT EXISTS idx_stock_instruments_fundamentals_updated
    ON stock_instruments(fundamentals_updated_at)
    WHERE is_active = TRUE;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON COLUMN stock_instruments.earnings_date IS 'Next earnings report date';
COMMENT ON COLUMN stock_instruments.earnings_date_estimated IS 'Whether earnings date is estimated vs confirmed';
COMMENT ON COLUMN stock_instruments.beta IS 'Stock volatility relative to market (1.0 = market)';
COMMENT ON COLUMN stock_instruments.short_ratio IS 'Days to cover short interest';
COMMENT ON COLUMN stock_instruments.short_percent_float IS 'Percentage of float shares that are short';
COMMENT ON COLUMN stock_instruments.institutional_percent IS 'Percentage owned by institutions';
COMMENT ON COLUMN stock_instruments.insider_percent IS 'Percentage owned by insiders';
COMMENT ON COLUMN stock_instruments.forward_pe IS 'Price/Earnings based on forward estimates';
COMMENT ON COLUMN stock_instruments.trailing_pe IS 'Price/Earnings based on trailing 12 months';
COMMENT ON COLUMN stock_instruments.peg_ratio IS 'P/E to Growth ratio (< 1 may indicate undervalued)';
COMMENT ON COLUMN stock_instruments.revenue_growth IS 'Year-over-year revenue growth rate';
COMMENT ON COLUMN stock_instruments.earnings_growth IS 'Year-over-year earnings growth rate';
COMMENT ON COLUMN stock_instruments.profit_margin IS 'Net profit margin percentage';
COMMENT ON COLUMN stock_instruments.return_on_equity IS 'Return on equity percentage';
COMMENT ON COLUMN stock_instruments.debt_to_equity IS 'Total debt / Total equity ratio';
COMMENT ON COLUMN stock_instruments.current_ratio IS 'Current assets / Current liabilities';
COMMENT ON COLUMN stock_instruments.dividend_yield IS 'Annual dividend yield percentage';
COMMENT ON COLUMN stock_instruments.analyst_rating IS 'Consensus analyst rating (buy/hold/sell)';
COMMENT ON COLUMN stock_instruments.fundamentals_updated_at IS 'Last time fundamentals were synced from yfinance';
