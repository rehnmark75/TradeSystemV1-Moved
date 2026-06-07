-- 034_create_stock_intraday_candles.sql
--
-- Persist the raw 5m bars the intraday VWAP/RVOL worker already downloads each
-- cycle for the ~50-name daytrade pool (9:25-10:30 ET window). Previously the
-- worker derived scalars into stock_intraday_state and discarded the bars; this
-- table keeps them as the substrate for future intraday-derived entry logic
-- (intraday ATR, VWAP-reclaim/hold, pullback-to-structure) and for measuring
-- whether the VWAP/RVOL entry gates actually help.
--
-- This table is also created idempotently by IntradayVwapSync.ensure_schema();
-- this file exists for schema-doc / DR parity. Lives in the `stocks` DB.
--
-- The PK makes re-fetching the same bars every ~2.5 min idempotent: the
-- still-forming last bar is overwritten in place (ON CONFLICT DO UPDATE), not
-- duplicated.

CREATE TABLE IF NOT EXISTS stock_intraday_candles (
    ticker     VARCHAR(20)      NOT NULL,
    timeframe  VARCHAR(10)      NOT NULL DEFAULT '5m',
    bar_time   TIMESTAMPTZ      NOT NULL,   -- bar START, tz-aware (ET)
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    volume     BIGINT,
    source     VARCHAR(20)      DEFAULT 'yfinance_5m',
    as_of      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, timeframe, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_stock_intraday_candles_ticker_time
    ON stock_intraday_candles(ticker, bar_time DESC);
