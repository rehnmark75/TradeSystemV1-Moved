-- Widen smc_rejection_outcomes.rejection_stage from varchar(20) to varchar(50)
-- Required because newer SMC stage names exceed 20 chars:
--   DIRECTION_QUALITY_GATE (22), EPIC_REGIME_HEALTH_GATE (23), CONFIDENCE_QUALITY_BAND (23)

DROP VIEW IF EXISTS v_smc_outcome_by_session;
DROP VIEW IF EXISTS v_smc_outcome_by_pair;
DROP VIEW IF EXISTS v_smc_missed_profit_analysis;
DROP VIEW IF EXISTS v_smc_outcome_by_stage;

ALTER TABLE smc_rejection_outcomes
    ALTER COLUMN rejection_stage TYPE character varying(50);

CREATE VIEW v_smc_outcome_by_stage AS
SELECT
    rejection_stage,
    COUNT(*) as total_analyzed,
    COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
    COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
    COUNT(CASE WHEN outcome = 'STILL_OPEN' THEN 1 END) as still_open,
    COUNT(CASE WHEN outcome = 'INSUFFICIENT_DATA' THEN 1 END) as insufficient_data,
    ROUND(
        COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
        NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
        1
    ) as would_be_win_rate,
    ROUND(SUM(potential_profit_pips)::numeric, 1) as net_potential_pips,
    ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
    ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
    ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe_pips,
    ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae_pips,
    ROUND(AVG(time_to_outcome_minutes)::numeric, 0) as avg_time_to_outcome_mins
FROM smc_rejection_outcomes
WHERE analysis_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY rejection_stage
ORDER BY total_analyzed DESC;

CREATE VIEW v_smc_missed_profit_analysis AS
SELECT
    pair,
    rejection_stage,
    COUNT(*) as total_analyzed,
    COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as missed_winners,
    COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as avoided_losers,
    ROUND(
        COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
        NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
        1
    ) as would_be_win_rate,
    ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
    ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
    ROUND(AVG(CASE WHEN outcome = 'HIT_TP' THEN time_to_outcome_minutes END)::numeric, 0) as avg_time_to_tp_mins
FROM smc_rejection_outcomes
WHERE analysis_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY pair, rejection_stage
ORDER BY missed_profit_pips DESC;

CREATE VIEW v_smc_outcome_by_pair AS
SELECT
    pair,
    COUNT(*) as total_analyzed,
    COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
    COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
    ROUND(
        COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
        NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
        1
    ) as would_be_win_rate,
    ROUND(SUM(potential_profit_pips)::numeric, 1) as net_potential_pips
FROM smc_rejection_outcomes
WHERE analysis_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY pair
ORDER BY net_potential_pips DESC;

CREATE VIEW v_smc_outcome_by_session AS
SELECT
    market_session,
    rejection_stage,
    COUNT(*) as total_analyzed,
    COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
    COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
    ROUND(
        COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
        NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
        1
    ) as would_be_win_rate,
    ROUND(SUM(potential_profit_pips)::numeric, 1) as net_potential_pips
FROM smc_rejection_outcomes
WHERE analysis_timestamp >= NOW() - INTERVAL '30 days'
  AND market_session IS NOT NULL
GROUP BY market_session, rejection_stage
ORDER BY market_session, total_analyzed DESC;
