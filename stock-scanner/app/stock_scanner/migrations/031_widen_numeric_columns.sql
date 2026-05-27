-- =============================================================================
-- Migration 031: Widen numeric columns to prevent overflow
-- Fixes: "numeric field overflow" errors for high-volatility tickers
-- (RGTI, AKTX, AAL with extreme ATR/perf %; NVDA/NOK with large volume metrics)
-- =============================================================================

-- Drop views that depend on columns being altered (recreated below)
DROP VIEW IF EXISTS stock_high_volume_movers;
DROP VIEW IF EXISTS stock_watchlist_summary;
DROP VIEW IF EXISTS v_sector_leaders;
DROP VIEW IF EXISTS v_active_scanner_signals;

-- stock_screening_metrics: widen NUMERIC(6,2) columns
-- Max was 9,999.99 — any ticker with ATR% or perf > ~9999% overflows
ALTER TABLE stock_screening_metrics
    ALTER COLUMN atr_percent TYPE NUMERIC(14,2),
    ALTER COLUMN historical_volatility_20 TYPE NUMERIC(14,2),
    ALTER COLUMN price_change_1d TYPE NUMERIC(14,2),
    ALTER COLUMN price_change_5d TYPE NUMERIC(14,2),
    ALTER COLUMN price_change_20d TYPE NUMERIC(14,2),
    ALTER COLUMN price_change_60d TYPE NUMERIC(14,2),
    ALTER COLUMN price_vs_sma20 TYPE NUMERIC(14,2),
    ALTER COLUMN price_vs_sma50 TYPE NUMERIC(14,2),
    ALTER COLUMN price_vs_sma200 TYPE NUMERIC(14,2),
    ALTER COLUMN daily_range_percent TYPE NUMERIC(14,2),
    ALTER COLUMN weekly_range_percent TYPE NUMERIC(14,2),
    ALTER COLUMN percentile_volume TYPE NUMERIC(14,2),
    ALTER COLUMN nearest_ob_distance TYPE NUMERIC(14,2),
    ALTER COLUMN momentum_10 TYPE NUMERIC(14,2),
    ALTER COLUMN cci_20 TYPE NUMERIC(14,2),
    ALTER COLUMN pct_from_52w_high TYPE NUMERIC(14,2),
    ALTER COLUMN pct_from_52w_low TYPE NUMERIC(14,2),
    ALTER COLUMN gap_percent TYPE NUMERIC(14,2),
    ALTER COLUMN perf_1w TYPE NUMERIC(14,2),
    ALTER COLUMN perf_1m TYPE NUMERIC(14,2),
    ALTER COLUMN perf_3m TYPE NUMERIC(14,2),
    ALTER COLUMN perf_6m TYPE NUMERIC(14,2),
    ALTER COLUMN perf_ytd TYPE NUMERIC(14,2),
    ALTER COLUMN adx TYPE NUMERIC(14,2),
    ALTER COLUMN avg_daily_change_5d TYPE NUMERIC(14,2),
    ALTER COLUMN stoch_k TYPE NUMERIC(14,2),
    ALTER COLUMN stoch_d TYPE NUMERIC(14,2),
    ALTER COLUMN stoch_rsi_k TYPE NUMERIC(14,2),
    ALTER COLUMN stoch_rsi_d TYPE NUMERIC(14,2),
    ALTER COLUMN williams_r TYPE NUMERIC(14,2),
    ALTER COLUMN ultimate_osc TYPE NUMERIC(14,2),
    ALTER COLUMN adx_14 TYPE NUMERIC(14,2),
    ALTER COLUMN plus_di TYPE NUMERIC(14,2),
    ALTER COLUMN minus_di TYPE NUMERIC(14,2);

-- stock_watchlist: widen all overflow-prone columns
ALTER TABLE stock_watchlist
    ALTER COLUMN atr_percent TYPE NUMERIC(14,2),
    ALTER COLUMN price_change_20d TYPE NUMERIC(14,2),
    ALTER COLUMN avg_daily_change_5d TYPE NUMERIC(14,2),
    ALTER COLUMN pct_from_52w_high TYPE NUMERIC(14,2);

-- Recreate views
CREATE OR REPLACE VIEW stock_high_volume_movers AS
SELECT
    m.ticker,
    i.name,
    m.current_price,
    m.atr_percent,
    m.avg_dollar_volume / 1000000 AS dollar_vol_millions,
    m.relative_volume,
    m.price_change_1d,
    m.price_change_5d,
    m.trend_strength,
    m.rsi_14,
    m.calculation_date
FROM stock_screening_metrics m
JOIN stock_instruments i ON m.ticker = i.ticker
WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
  AND m.avg_dollar_volume >= 10000000
  AND m.atr_percent >= 2.0
  AND m.relative_volume >= 1.0
ORDER BY m.avg_dollar_volume DESC, m.atr_percent DESC
LIMIT 100;

CREATE OR REPLACE VIEW v_sector_leaders AS
SELECT
    m.ticker,
    i.name,
    i.sector,
    m.current_price,
    m.rs_vs_spy,
    m.rs_percentile,
    m.rs_trend,
    s.rs_vs_spy AS sector_rs,
    s.rs_percentile AS sector_rs_percentile,
    s.sector_stage,
    m.trend_strength,
    m.ma_alignment,
    m.atr_percent,
    m.price_change_20d
FROM stock_screening_metrics m
JOIN stock_instruments i ON m.ticker = i.ticker
LEFT JOIN sector_analysis s ON i.sector = s.sector AND s.calculation_date = m.calculation_date
WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
  AND m.rs_percentile >= 70
  AND s.rs_percentile >= 50
ORDER BY m.rs_percentile DESC, s.rs_percentile DESC;

CREATE OR REPLACE VIEW v_active_scanner_signals AS
SELECT
    s.id,
    s.signal_timestamp,
    s.scanner_name,
    s.ticker,
    s.signal_type,
    s.entry_price,
    s.stop_loss,
    s.take_profit_1,
    s.take_profit_2,
    s.risk_reward_ratio,
    s.risk_percent,
    s.composite_score,
    s.quality_tier,
    s.trend_score,
    s.momentum_score,
    s.volume_score,
    s.pattern_score,
    s.confluence_score,
    s.setup_description,
    s.confluence_factors,
    s.timeframe,
    s.market_regime,
    s.suggested_position_size_pct,
    s.max_risk_per_trade_pct,
    s.status,
    s.trigger_timestamp,
    s.close_timestamp,
    s.close_price,
    s.realized_pnl_pct,
    s.realized_r_multiple,
    s.exit_reason,
    s.created_at,
    s.updated_at,
    i.name AS company_name,
    w.tier,
    w.score AS watchlist_score,
    w.relative_volume,
    w.atr_percent
FROM stock_scanner_signals s
JOIN stock_instruments i ON s.ticker = i.ticker
LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
    AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
WHERE s.status = 'active'
ORDER BY s.composite_score DESC;

CREATE OR REPLACE VIEW stock_watchlist_summary AS
SELECT
    tier,
    COUNT(*) AS stock_count,
    ROUND(AVG(score), 1) AS avg_score,
    ROUND(AVG(atr_percent), 2) AS avg_atr,
    ROUND(AVG(avg_dollar_volume) / 1000000, 1) AS avg_dollar_vol_m,
    ROUND(AVG(price_change_20d), 2) AS avg_20d_change
FROM stock_watchlist
WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
GROUP BY tier
ORDER BY tier;
