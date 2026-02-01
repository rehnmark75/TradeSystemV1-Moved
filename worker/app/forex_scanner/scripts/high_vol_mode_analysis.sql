-- HIGH_VOL_MODE Analysis Query
-- Run after 1-2 weeks of data collection to validate filter effectiveness
-- before implementing actual filtering behavior

-- Query 1: Compare predicted filter outcomes with actual trade results
-- This shows if the would_pass_filter prediction correlates with profitability
SELECT
    (strategy_metadata->'high_vol_analysis'->>'would_pass_filter')::boolean as would_pass,
    COUNT(*) as trades,
    SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN tl.profit_loss <= 0 THEN 1 ELSE 0 END) as losses,
    ROUND(100.0 * SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
    ROUND(AVG(tl.profit_loss)::numeric, 2) as avg_pnl,
    ROUND(SUM(tl.profit_loss)::numeric, 2) as total_pnl
FROM alert_history ah
JOIN trade_log tl ON tl.symbol = ah.epic
    AND tl.timestamp BETWEEN ah.alert_timestamp - interval '5 min'
    AND ah.alert_timestamp + interval '5 min'
WHERE ah.strategy_metadata->'high_vol_analysis' IS NOT NULL
    AND tl.status = 'closed'
GROUP BY 1
ORDER BY 1 DESC;

-- Query 2: Breakdown by pass_reason
-- Shows which specific criteria (high_volume, american_session) are most predictive
SELECT
    strategy_metadata->'high_vol_analysis'->>'pass_reason' as pass_reason,
    COUNT(*) as trades,
    SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
    ROUND(AVG(tl.profit_loss)::numeric, 2) as avg_pnl
FROM alert_history ah
JOIN trade_log tl ON tl.symbol = ah.epic
    AND tl.timestamp BETWEEN ah.alert_timestamp - interval '5 min'
    AND ah.alert_timestamp + interval '5 min'
WHERE ah.strategy_metadata->'high_vol_analysis' IS NOT NULL
    AND tl.status = 'closed'
GROUP BY 1
ORDER BY trades DESC;

-- Query 3: Volume ratio bucket analysis (validate the 1.2 threshold)
SELECT
    CASE
        WHEN (strategy_metadata->'high_vol_analysis'->>'volume_ratio')::numeric >= 1.5 THEN '>=1.5 (very high)'
        WHEN (strategy_metadata->'high_vol_analysis'->>'volume_ratio')::numeric >= 1.2 THEN '1.2-1.5 (high)'
        WHEN (strategy_metadata->'high_vol_analysis'->>'volume_ratio')::numeric >= 0.8 THEN '0.8-1.2 (normal)'
        ELSE '<0.8 (low)'
    END as volume_bucket,
    COUNT(*) as trades,
    SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
    ROUND(AVG(tl.profit_loss)::numeric, 2) as avg_pnl
FROM alert_history ah
JOIN trade_log tl ON tl.symbol = ah.epic
    AND tl.timestamp BETWEEN ah.alert_timestamp - interval '5 min'
    AND ah.alert_timestamp + interval '5 min'
WHERE ah.strategy_metadata->'high_vol_analysis' IS NOT NULL
    AND tl.status = 'closed'
GROUP BY 1
ORDER BY 1;

-- Query 4: Session performance during high volatility
SELECT
    strategy_metadata->'high_vol_analysis'->>'session' as session,
    COUNT(*) as trades,
    SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
    ROUND(AVG(tl.profit_loss)::numeric, 2) as avg_pnl
FROM alert_history ah
JOIN trade_log tl ON tl.symbol = ah.epic
    AND tl.timestamp BETWEEN ah.alert_timestamp - interval '5 min'
    AND ah.alert_timestamp + interval '5 min'
WHERE ah.strategy_metadata->'high_vol_analysis' IS NOT NULL
    AND tl.status = 'closed'
GROUP BY 1
ORDER BY trades DESC;

-- Query 5: Confidence threshold analysis (validate 0.55 threshold)
SELECT
    CASE
        WHEN (strategy_metadata->'high_vol_analysis'->>'confidence')::numeric >= 0.65 THEN '>=0.65 (high)'
        WHEN (strategy_metadata->'high_vol_analysis'->>'confidence')::numeric >= 0.55 THEN '0.55-0.65 (medium)'
        WHEN (strategy_metadata->'high_vol_analysis'->>'confidence')::numeric >= 0.45 THEN '0.45-0.55 (low)'
        ELSE '<0.45 (very low)'
    END as confidence_bucket,
    COUNT(*) as trades,
    SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
    ROUND(AVG(tl.profit_loss)::numeric, 2) as avg_pnl
FROM alert_history ah
JOIN trade_log tl ON tl.symbol = ah.epic
    AND tl.timestamp BETWEEN ah.alert_timestamp - interval '5 min'
    AND ah.alert_timestamp + interval '5 min'
WHERE ah.strategy_metadata->'high_vol_analysis' IS NOT NULL
    AND tl.status = 'closed'
GROUP BY 1
ORDER BY 1;

-- Query 6: Check if high_vol_analysis is being stored (verification query)
SELECT
    DATE(alert_timestamp) as date,
    COUNT(*) as total_alerts,
    COUNT(strategy_metadata->'high_vol_analysis') as with_analysis,
    strategy_metadata->'high_vol_analysis'->>'regime' as regime
FROM alert_history
WHERE strategy_metadata->'high_vol_analysis' IS NOT NULL
    AND alert_timestamp > NOW() - INTERVAL '7 days'
GROUP BY 1, 4
ORDER BY 1 DESC;

-- Query 7: Sample of recent high_vol_analysis data
SELECT
    alert_timestamp,
    epic,
    signal_type,
    strategy_metadata->'high_vol_analysis' as high_vol_analysis
FROM alert_history
WHERE strategy_metadata->'high_vol_analysis' IS NOT NULL
ORDER BY alert_timestamp DESC
LIMIT 10;
