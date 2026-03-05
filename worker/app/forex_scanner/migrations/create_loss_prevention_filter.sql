-- Loss Prevention Filter (LPF) - Pattern-Based Trade Blocking
-- Creates tables in strategy_config database for configurable rule-based filtering
-- Deploys in monitor mode (logs decisions without blocking)

-- Part 1: Add LPF columns to alert_history in forex database
\c forex;

ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS lpf_penalty NUMERIC(5,2);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS lpf_would_block BOOLEAN DEFAULT FALSE;

-- Part 2: Create LPF tables in strategy_config database
-- Connect to strategy_config database
\c strategy_config;

-- ============================================================
-- Table 1: loss_prevention_rules
-- Configurable rules with JSONB condition definitions
-- ============================================================
CREATE TABLE IF NOT EXISTS loss_prevention_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL UNIQUE,
    category CHAR(1) NOT NULL,  -- A=pair, B=confidence, C=time, D=regime, E=technical, F=boost
    description TEXT,
    penalty NUMERIC(4,2) NOT NULL,  -- Positive = penalty, Negative = boost
    condition_config JSONB NOT NULL,  -- Flexible rule conditions
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    apply_in_backtest BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- Table 2: loss_prevention_config
-- Global settings (threshold, mode, enabled)
-- ============================================================
CREATE TABLE IF NOT EXISTS loss_prevention_config (
    id SERIAL PRIMARY KEY,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    block_mode VARCHAR(20) NOT NULL DEFAULT 'monitor',  -- 'monitor' or 'block'
    penalty_threshold NUMERIC(4,2) NOT NULL DEFAULT 0.60,
    apply_in_backtest BOOLEAN NOT NULL DEFAULT TRUE,
    log_decisions BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- Table 3: loss_prevention_decisions
-- Decision log for analysis and tuning
-- ============================================================
CREATE TABLE IF NOT EXISTS loss_prevention_decisions (
    id SERIAL PRIMARY KEY,
    alert_id INTEGER,  -- FK to alert_history (nullable for monitor mode)
    epic VARCHAR(100),
    signal_type VARCHAR(20),
    confidence NUMERIC(4,2),
    total_penalty NUMERIC(5,2),
    triggered_rules JSONB,  -- Array of {rule_name, category, penalty}
    decision VARCHAR(20) NOT NULL,  -- 'allowed', 'would_block', 'blocked'
    signal_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lpf_decisions_created ON loss_prevention_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_lpf_decisions_decision ON loss_prevention_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_lpf_decisions_epic ON loss_prevention_decisions(epic);

-- ============================================================
-- Insert default config (monitor mode)
-- ============================================================
INSERT INTO loss_prevention_config (is_enabled, block_mode, penalty_threshold, apply_in_backtest, log_decisions)
VALUES (TRUE, 'monitor', 0.60, TRUE, TRUE)
ON CONFLICT DO NOTHING;

-- ============================================================
-- Insert default rules
-- ============================================================

-- Category A: Pair-specific rules
INSERT INTO loss_prevention_rules (rule_name, category, description, penalty, condition_config) VALUES
('usdjpy_block', 'A', 'USDJPY base penalty - 23.1% WR overall', 0.45,
 '{"type": "pair", "epic_contains": "USDJPY"}'),
('usdjpy_high_conf', 'A', 'USDJPY + high confidence - 0% WR at conf>=0.65', 0.55,
 '{"type": "pair_and_confidence", "epic_contains": "USDJPY", "min_confidence": 0.65}'),
('audjpy_low_vol', 'A', 'AUDJPY in low volatility - 25% WR', 0.40,
 '{"type": "pair_and_regime", "epic_contains": "AUDJPY", "regime": "low_volatility"}'),
('audjpy_bad_hours', 'A', 'AUDJPY at bad hours - 0% WR', 0.35,
 '{"type": "pair_and_hours", "epic_contains": "AUDJPY", "hours": [12, 19, 20, 21, 22]}')
ON CONFLICT (rule_name) DO NOTHING;

-- Category B: Confidence rules
INSERT INTO loss_prevention_rules (rule_name, category, description, penalty, condition_config) VALUES
('extreme_confidence', 'B', 'Confidence >= 0.70 inversely predictive - 40% WR, -449 PnL', 0.35,
 '{"type": "confidence_range", "min_confidence": 0.70}'),
('high_confidence', 'B', 'Confidence 0.65-0.70 below average WR', 0.20,
 '{"type": "confidence_range", "min_confidence": 0.65, "max_confidence": 0.70}'),
('conf_quality_combo', 'B', 'Triple high-score inversion pattern', 0.30,
 '{"type": "multi_threshold", "conditions": {"confidence": 0.65, "entry_quality_score": 0.70, "htf_bias_score": 0.70}}')
ON CONFLICT (rule_name) DO NOTHING;

-- Category C: Time rules
INSERT INTO loss_prevention_rules (rule_name, category, description, penalty, condition_config) VALUES
('wednesday_block', 'C', 'Wednesday - 31.3% WR', 0.25,
 '{"type": "day_of_week", "days": [2]}'),
('bad_hours', 'C', 'Loss concentration hours', 0.20,
 '{"type": "hour_utc", "hours": [8, 15, 16, 17, 22]}')
ON CONFLICT (rule_name) DO NOTHING;

-- Category D: Regime rules
INSERT INTO loss_prevention_rules (rule_name, category, description, penalty, condition_config) VALUES
('low_volatility', 'D', 'Low volatility regime - 48% WR, -688 PnL', 0.30,
 '{"type": "regime", "regime": "low_volatility"}'),
('trending_misaligned', 'D', 'Trending but not aligned - negative PnL', 0.20,
 '{"type": "regime_and_alignment", "regime": "trending", "all_timeframes_aligned": false}')
ON CONFLICT (rule_name) DO NOTHING;

-- Category E: Technical indicator rules
INSERT INTO loss_prevention_rules (rule_name, category, description, penalty, condition_config) VALUES
('low_rsi_buy', 'E', 'BUY with RSI < 45 - winners avg 53 vs losers 48', 0.15,
 '{"type": "direction_and_indicator", "direction": "BUY", "indicator": "rsi", "max_value": 45}'),
('low_adx', 'E', 'ADX < 18 - weak trend', 0.10,
 '{"type": "indicator_threshold", "indicator": "adx", "max_value": 18}'),
('low_mtf_confluence', 'E', 'MTF confluence < 0.50 - winners 0.56 vs losers 0.54', 0.15,
 '{"type": "indicator_threshold", "indicator": "mtf_confluence_score", "max_value": 0.50}')
ON CONFLICT (rule_name) DO NOTHING;

-- Category F: Boost rules (negative penalties)
INSERT INTO loss_prevention_rules (rule_name, category, description, penalty, condition_config) VALUES
('sweet_spot_conf', 'F', 'Confidence sweet spot 0.55-0.60 - 80% WR', -0.15,
 '{"type": "confidence_range", "min_confidence": 0.55, "max_confidence": 0.60}'),
('trending_aligned', 'F', 'Trending + all timeframes aligned - 100% WR', -0.20,
 '{"type": "regime_and_alignment", "regime": "trending", "all_timeframes_aligned": true}'),
('asian_session', 'F', 'Asian session hours 23-07 UTC - only profitable session', -0.10,
 '{"type": "hour_utc", "hours": [23, 0, 1, 2, 3, 4, 5, 6, 7]}')
ON CONFLICT (rule_name) DO NOTHING;
