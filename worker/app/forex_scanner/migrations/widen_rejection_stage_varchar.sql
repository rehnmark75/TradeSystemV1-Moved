-- Widen rejection_stage column from varchar(20) to varchar(50)
-- Required because newer stage names exceed 20 chars:
--   DIRECTION_QUALITY_GATE (22), EPIC_REGIME_HEALTH_GATE (23), CONFIDENCE_QUALITY_BAND (23)

-- Drop dependent views first
DROP VIEW IF EXISTS v_rejection_analysis_by_conditions;
DROP VIEW IF EXISTS v_smc_rejection_by_stage;
DROP VIEW IF EXISTS v_smc_rejection_by_session;
DROP VIEW IF EXISTS v_smc_rejection_by_hour;
DROP VIEW IF EXISTS v_smc_near_misses;

ALTER TABLE smc_simple_rejections ALTER COLUMN rejection_stage TYPE character varying(50);

-- Recreate views
CREATE VIEW v_rejection_analysis_by_conditions AS
  SELECT rejection_stage, market_regime_detected AS regime, volatility_state,
     count(*) AS rejection_count,
     round(avg(efficiency_ratio), 4) AS avg_er,
     round(avg(confidence_score), 3) AS avg_confidence_at_rejection,
     round(avg(adx_value), 1) AS avg_adx
    FROM smc_simple_rejections
   WHERE scan_timestamp >= (now() - '30 days'::interval) AND market_regime_detected IS NOT NULL
   GROUP BY rejection_stage, market_regime_detected, volatility_state
   ORDER BY count(*) DESC;

CREATE VIEW v_smc_rejection_by_stage AS
  SELECT rejection_stage,
     count(*) AS rejection_count,
     count(DISTINCT epic) AS unique_pairs,
     round(avg(atr_percentile), 2) AS avg_atr_percentile,
     round(avg(spread_pips), 2) AS avg_spread
    FROM smc_simple_rejections
   WHERE scan_timestamp >= (now() - '30 days'::interval)
   GROUP BY rejection_stage
   ORDER BY count(*) DESC;

CREATE VIEW v_smc_rejection_by_session AS
  SELECT market_session, rejection_stage,
     count(*) AS rejection_count,
     round(avg(atr_percentile), 2) AS avg_volatility
    FROM smc_simple_rejections
   WHERE scan_timestamp >= (now() - '30 days'::interval)
   GROUP BY market_session, rejection_stage
   ORDER BY market_session, count(*) DESC;

CREATE VIEW v_smc_rejection_by_hour AS
  SELECT market_hour, rejection_stage,
     count(*) AS rejection_count
    FROM smc_simple_rejections
   WHERE scan_timestamp >= (now() - '30 days'::interval)
   GROUP BY market_hour, rejection_stage
   ORDER BY market_hour;

CREATE VIEW v_smc_near_misses AS
  SELECT scan_timestamp, epic, pair, attempted_direction, confidence_score,
     rejection_reason, potential_rr_ratio, market_session,
     ema_distance_pips, pullback_depth, fib_zone
    FROM smc_simple_rejections
   WHERE rejection_stage::text = 'CONFIDENCE'::text AND scan_timestamp >= (now() - '30 days'::interval)
   ORDER BY confidence_score DESC;
