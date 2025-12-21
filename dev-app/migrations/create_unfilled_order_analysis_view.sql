-- ============================================================================
-- UNFILLED ORDER ANALYSIS VIEW
-- ============================================================================
-- Purpose: Analyze unfilled stop-entry orders to determine if they were
--          "near misses" (would have been profitable) or "bad signals"
-- Created: 2025-12-19
--
-- This view answers:
-- 1. Would the order have filled if we waited longer? (4h, 24h)
-- 2. If filled, would it have hit TP or SL?
-- 3. Classification: GOOD_SIGNAL, BAD_SIGNAL, or INCONCLUSIVE
-- ============================================================================

DROP VIEW IF EXISTS v_unfilled_order_analysis;

CREATE VIEW v_unfilled_order_analysis AS
WITH unfilled_orders AS (
    SELECT
        t.id,
        t.symbol,
        t.direction,
        t.entry_price,
        t.sl_price,
        t.limit_price as tp_price,  -- Note: limit_price = TP level
        t.timestamp as order_time,
        t.monitor_until as expiry_time,
        t.alert_id,
        CASE WHEN t.symbol LIKE '%JPY%' THEN 0.01 ELSE 0.0001 END as pip_val
    FROM trade_log t
    WHERE t.status = 'limit_not_filled'
      AND t.symbol NOT LIKE '%CEEM%'  -- Exclude CEEM pairs (abnormal pricing)
),
price_analysis AS (
    SELECT
        o.*,
        -- 4-hour window after expiry
        (SELECT MIN(low) FROM ig_candles WHERE epic = o.symbol AND timeframe = 5
         AND start_time > o.expiry_time AND start_time <= o.expiry_time + INTERVAL '4 hours') as min_low_4h,
        (SELECT MAX(high) FROM ig_candles WHERE epic = o.symbol AND timeframe = 5
         AND start_time > o.expiry_time AND start_time <= o.expiry_time + INTERVAL '4 hours') as max_high_4h,
        -- 24-hour window after expiry
        (SELECT MIN(low) FROM ig_candles WHERE epic = o.symbol AND timeframe = 5
         AND start_time > o.expiry_time AND start_time <= o.expiry_time + INTERVAL '24 hours') as min_low_24h,
        (SELECT MAX(high) FROM ig_candles WHERE epic = o.symbol AND timeframe = 5
         AND start_time > o.expiry_time AND start_time <= o.expiry_time + INTERVAL '24 hours') as max_high_24h,
        -- Price at expiry
        (SELECT close FROM ig_candles WHERE epic = o.symbol AND timeframe = 5
         AND start_time > o.expiry_time ORDER BY start_time LIMIT 1) as price_at_expiry
    FROM unfilled_orders o
)
SELECT
    id,
    symbol,
    direction,
    order_time,
    expiry_time,
    alert_id,

    -- Price levels
    ROUND(entry_price::numeric, 5) as entry_level,
    ROUND(sl_price::numeric, 5) as stop_loss,
    ROUND(tp_price::numeric, 5) as take_profit,
    ROUND(price_at_expiry::numeric, 5) as price_at_expiry,

    -- Gap to entry at expiry (how close we were to filling)
    ROUND((ABS(entry_price - price_at_expiry) / pip_val)::numeric, 1) as gap_to_entry_pips,

    -- 4-hour analysis
    CASE
        WHEN direction = 'BUY' AND max_high_4h >= entry_price THEN TRUE
        WHEN direction = 'SELL' AND min_low_4h <= entry_price THEN TRUE
        ELSE FALSE
    END as would_fill_4h,

    CASE
        WHEN direction = 'BUY' AND max_high_4h >= tp_price THEN 'TP_HIT'
        WHEN direction = 'SELL' AND min_low_4h <= tp_price THEN 'TP_HIT'
        WHEN direction = 'BUY' AND min_low_4h <= sl_price THEN 'SL_HIT'
        WHEN direction = 'SELL' AND max_high_4h >= sl_price THEN 'SL_HIT'
        ELSE 'NEITHER'
    END as outcome_4h,

    -- 24-hour analysis
    CASE
        WHEN direction = 'BUY' AND max_high_24h >= entry_price THEN TRUE
        WHEN direction = 'SELL' AND min_low_24h <= entry_price THEN TRUE
        ELSE FALSE
    END as would_fill_24h,

    CASE
        WHEN direction = 'BUY' AND max_high_24h >= tp_price THEN 'TP_HIT'
        WHEN direction = 'SELL' AND min_low_24h <= tp_price THEN 'TP_HIT'
        WHEN direction = 'BUY' AND min_low_24h <= sl_price THEN 'SL_HIT'
        WHEN direction = 'SELL' AND max_high_24h >= sl_price THEN 'SL_HIT'
        ELSE 'NEITHER'
    END as outcome_24h,

    -- Signal quality classification
    CASE
        -- GOOD: Would have filled and hit TP
        WHEN (direction = 'BUY' AND max_high_24h >= tp_price) OR
             (direction = 'SELL' AND min_low_24h <= tp_price) THEN 'GOOD_SIGNAL'
        -- BAD: Would have filled and hit SL (or SL hit before fill)
        WHEN (direction = 'BUY' AND min_low_24h <= sl_price) OR
             (direction = 'SELL' AND max_high_24h >= sl_price) THEN 'BAD_SIGNAL'
        -- INCONCLUSIVE: Neither TP nor SL hit in 24h
        ELSE 'INCONCLUSIVE'
    END as signal_quality,

    -- Maximum favorable excursion (how far price moved in our direction)
    CASE
        WHEN direction = 'BUY' THEN ROUND(((max_high_24h - price_at_expiry) / pip_val)::numeric, 1)
        ELSE ROUND(((price_at_expiry - min_low_24h) / pip_val)::numeric, 1)
    END as max_favorable_pips,

    -- Maximum adverse excursion (how far price moved against us)
    CASE
        WHEN direction = 'BUY' THEN ROUND(((price_at_expiry - min_low_24h) / pip_val)::numeric, 1)
        ELSE ROUND(((max_high_24h - price_at_expiry) / pip_val)::numeric, 1)
    END as max_adverse_pips

FROM price_analysis;

-- Summary view for quick stats
DROP VIEW IF EXISTS v_unfilled_order_summary;

CREATE VIEW v_unfilled_order_summary AS
SELECT
    COUNT(*) as total_unfilled,
    SUM(CASE WHEN would_fill_4h THEN 1 ELSE 0 END) as would_fill_4h,
    SUM(CASE WHEN would_fill_24h THEN 1 ELSE 0 END) as would_fill_24h,
    SUM(CASE WHEN signal_quality = 'GOOD_SIGNAL' THEN 1 ELSE 0 END) as good_signals,
    SUM(CASE WHEN signal_quality = 'BAD_SIGNAL' THEN 1 ELSE 0 END) as bad_signals,
    SUM(CASE WHEN signal_quality = 'INCONCLUSIVE' THEN 1 ELSE 0 END) as inconclusive,
    ROUND(100.0 * SUM(CASE WHEN signal_quality = 'GOOD_SIGNAL' THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN signal_quality IN ('GOOD_SIGNAL', 'BAD_SIGNAL') THEN 1 ELSE 0 END), 0), 1) as win_rate_pct,
    ROUND(AVG(gap_to_entry_pips)::numeric, 1) as avg_gap_to_entry_pips,
    ROUND(AVG(max_favorable_pips)::numeric, 1) as avg_favorable_move_pips,
    ROUND(AVG(max_adverse_pips)::numeric, 1) as avg_adverse_move_pips
FROM v_unfilled_order_analysis;

-- Comments
COMMENT ON VIEW v_unfilled_order_analysis IS
'Analyzes unfilled stop-entry orders to determine signal quality. Shows if price would have reached entry, and whether trade would have been profitable.';

COMMENT ON VIEW v_unfilled_order_summary IS
'Summary statistics for unfilled order analysis - shows overall signal quality and win rate.';
