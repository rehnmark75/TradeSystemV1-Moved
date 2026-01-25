-- Finnhub analyst recommendation trends
CREATE TABLE IF NOT EXISTS stock_analyst_recommendations (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    period DATE NOT NULL,
    strong_buy INTEGER,
    buy INTEGER,
    hold INTEGER,
    sell INTEGER,
    strong_sell INTEGER,
    source VARCHAR(32) DEFAULT 'finnhub',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unique_recommendation_period UNIQUE (ticker, period)
);

CREATE INDEX IF NOT EXISTS idx_analyst_reco_ticker_period
    ON stock_analyst_recommendations (ticker, period DESC);

COMMENT ON TABLE stock_analyst_recommendations IS 'Analyst recommendation trends from Finnhub';
