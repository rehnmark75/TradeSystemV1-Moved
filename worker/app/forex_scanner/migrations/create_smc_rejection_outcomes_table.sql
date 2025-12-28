-- ============================================================================
-- SMC REJECTION OUTCOMES TABLE
-- ============================================================================
-- Purpose: Store outcome analysis for rejected SMC Simple signals
-- Tracks whether rejected signals would have been profitable
-- Created: 2025-12-28
-- Storage: ~50-100 KB/day (depends on rejection volume)
-- ============================================================================

-- Drop existing objects if they exist (for clean migration)
DROP VIEW IF EXISTS v_smc_outcome_by_session;
DROP VIEW IF EXISTS v_smc_outcome_by_pair;
DROP VIEW IF EXISTS v_smc_missed_profit_analysis;
DROP VIEW IF EXISTS v_smc_outcome_by_stage;
DROP TABLE IF EXISTS smc_rejection_outcomes;

-- ============================================================================
-- MAIN TABLE
-- ============================================================================

CREATE TABLE smc_rejection_outcomes (
    id SERIAL PRIMARY KEY,

    -- Link to original rejection
    rejection_id INTEGER NOT NULL,
    epic VARCHAR(50) NOT NULL,
    pair VARCHAR(20) NOT NULL,
    rejection_timestamp TIMESTAMP NOT NULL,
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Original rejection context (denormalized for query performance)
    rejection_stage VARCHAR(20) NOT NULL,
    attempted_direction VARCHAR(10) NOT NULL,  -- 'BULL' or 'BEAR'
    market_session VARCHAR(20),
    market_hour INTEGER,

    -- Entry parameters (calculated from rejection data)
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss_price DECIMAL(10,5) NOT NULL,
    take_profit_price DECIMAL(10,5) NOT NULL,
    spread_at_rejection DECIMAL(5,2),

    -- Outcome determination
    outcome VARCHAR(20) NOT NULL,  -- 'HIT_TP', 'HIT_SL', 'STILL_OPEN', 'INSUFFICIENT_DATA'
    outcome_price DECIMAL(10,5),    -- Price at which TP or SL was hit
    outcome_timestamp TIMESTAMP,    -- When TP or SL was hit

    -- Time metrics
    time_to_outcome_minutes INTEGER,  -- Time from entry to outcome
    time_to_mfe_minutes INTEGER,      -- Time to maximum favorable excursion
    time_to_mae_minutes INTEGER,      -- Time to maximum adverse excursion

    -- Excursion analysis (MFE/MAE)
    max_favorable_excursion_pips DECIMAL(8,2),  -- Best the trade got
    max_adverse_excursion_pips DECIMAL(8,2),    -- Worst drawdown
    mfe_timestamp TIMESTAMP,
    mae_timestamp TIMESTAMP,

    -- Profit/loss calculation (in pips)
    potential_profit_pips DECIMAL(8,2),  -- If HIT_TP = +15, if HIT_SL = -9
    risk_reward_realized DECIMAL(6,3),   -- Actual R:R achieved

    -- Analysis quality
    candle_count_analyzed INTEGER,       -- Number of candles in analysis window
    data_quality_score DECIMAL(3,2),     -- 0-1 quality score
    analysis_notes TEXT,                 -- Any issues or notes

    -- Configuration used for analysis
    fixed_sl_pips DECIMAL(5,2) DEFAULT 9,   -- SL used for analysis
    fixed_tp_pips DECIMAL(5,2) DEFAULT 15,  -- TP used for analysis
    analysis_version VARCHAR(10) DEFAULT '1.0',

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT fk_smc_rejection FOREIGN KEY (rejection_id)
        REFERENCES smc_simple_rejections(id) ON DELETE CASCADE,
    CONSTRAINT chk_outcome CHECK (outcome IN ('HIT_TP', 'HIT_SL', 'STILL_OPEN', 'INSUFFICIENT_DATA')),
    CONSTRAINT chk_direction CHECK (attempted_direction IN ('BULL', 'BEAR'))
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Primary query patterns
CREATE INDEX idx_smc_outcome_rejection_id ON smc_rejection_outcomes(rejection_id);
CREATE INDEX idx_smc_outcome_epic ON smc_rejection_outcomes(epic);
CREATE INDEX idx_smc_outcome_pair ON smc_rejection_outcomes(pair);
CREATE INDEX idx_smc_outcome_outcome ON smc_rejection_outcomes(outcome);
CREATE INDEX idx_smc_outcome_stage ON smc_rejection_outcomes(rejection_stage);
CREATE INDEX idx_smc_outcome_direction ON smc_rejection_outcomes(attempted_direction);
CREATE INDEX idx_smc_outcome_session ON smc_rejection_outcomes(market_session);

-- Time-based queries
CREATE INDEX idx_smc_outcome_rejection_ts ON smc_rejection_outcomes(rejection_timestamp);
CREATE INDEX idx_smc_outcome_analysis_ts ON smc_rejection_outcomes(analysis_timestamp);

-- Performance analysis
CREATE INDEX idx_smc_outcome_profit ON smc_rejection_outcomes(potential_profit_pips);
CREATE INDEX idx_smc_outcome_mfe ON smc_rejection_outcomes(max_favorable_excursion_pips);

-- Composite indexes for common queries
CREATE INDEX idx_smc_outcome_stage_outcome ON smc_rejection_outcomes(rejection_stage, outcome);
CREATE INDEX idx_smc_outcome_epic_ts ON smc_rejection_outcomes(epic, rejection_timestamp DESC);
CREATE INDEX idx_smc_outcome_pair_stage ON smc_rejection_outcomes(pair, rejection_stage);

-- ============================================================================
-- ANALYSIS VIEWS
-- ============================================================================

-- View: Win rate by rejection stage (primary analysis view)
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

-- View: Missed profit analysis by pair and stage
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

-- View: Outcome by pair (overall pair performance)
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

-- View: Outcome by session
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

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE smc_rejection_outcomes IS 'Stores outcome analysis for SMC Simple rejected signals - determines if they would have been profitable with SL=9, TP=15';
COMMENT ON COLUMN smc_rejection_outcomes.outcome IS 'Trade outcome: HIT_TP (would be winner), HIT_SL (would be loser), STILL_OPEN (not resolved), INSUFFICIENT_DATA (not enough candles)';
COMMENT ON COLUMN smc_rejection_outcomes.max_favorable_excursion_pips IS 'Maximum favorable price movement from entry in pips (best the trade got)';
COMMENT ON COLUMN smc_rejection_outcomes.max_adverse_excursion_pips IS 'Maximum adverse price movement from entry in pips (worst drawdown)';
COMMENT ON COLUMN smc_rejection_outcomes.potential_profit_pips IS 'Profit in pips: +15 for HIT_TP, -9 for HIT_SL';
COMMENT ON COLUMN smc_rejection_outcomes.data_quality_score IS 'Quality score 0-1 based on candle count and data gaps';

-- ============================================================================
-- GRANT PERMISSIONS (adjust as needed for your setup)
-- ============================================================================
-- GRANT SELECT, INSERT, UPDATE ON smc_rejection_outcomes TO trading_app;
-- GRANT SELECT ON v_smc_outcome_by_stage TO trading_app;
-- GRANT SELECT ON v_smc_missed_profit_analysis TO trading_app;
-- GRANT SELECT ON v_smc_outcome_by_pair TO trading_app;
-- GRANT SELECT ON v_smc_outcome_by_session TO trading_app;
