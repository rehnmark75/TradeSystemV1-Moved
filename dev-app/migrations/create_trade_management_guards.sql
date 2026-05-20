-- Trade Management Guards
-- Post-entry risk management rules evaluated each monitoring cycle.
-- Sibling to loss_prevention_rules (which guards entries).
--
-- Deployment path:
--   1. Insert guard with mode='monitor'  → decisions logged, no closes
--   2. After review: UPDATE trade_management_guards SET mode='active' WHERE name=...
--   3. decisions.executed column distinguishes monitor vs real close records

CREATE TABLE IF NOT EXISTS trade_management_guards (
    id                  SERIAL PRIMARY KEY,
    name                TEXT UNIQUE NOT NULL,           -- 'failed_followthrough_eurusd_buy'
    guard_type          TEXT NOT NULL,                  -- 'failed_followthrough' | future types
    enabled             BOOLEAN NOT NULL DEFAULT TRUE,
    mode                TEXT NOT NULL DEFAULT 'monitor', -- 'monitor' | 'active'
    applies_to_strategies TEXT[],                       -- NULL = all strategies
    epic_filter         TEXT[],                         -- NULL = all epics
    direction_filter    TEXT[],                         -- NULL = all directions
    condition_config    JSONB NOT NULL DEFAULT '{}',    -- guard-type-specific params
    notes               TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trade_management_decisions (
    id                  SERIAL PRIMARY KEY,
    guard_id            INT NOT NULL REFERENCES trade_management_guards(id),
    trade_id            INT NOT NULL,                   -- trade_log.id in forex DB
    deal_id             TEXT,
    epic                TEXT,
    direction           TEXT,
    strategy            TEXT,
    fired_at            TIMESTAMP NOT NULL DEFAULT NOW(),
    mode_at_firing      TEXT NOT NULL,                  -- captures monitor vs active
    age_minutes         FLOAT,
    mfe_pips            FLOAT,
    mae_pips            FLOAT,
    executed            BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE only when mode='active' and close succeeded
    close_error         TEXT                            -- populated when executed=FALSE and mode='active'
);

CREATE INDEX IF NOT EXISTS idx_tmg_enabled ON trade_management_guards(enabled);
CREATE INDEX IF NOT EXISTS idx_tmd_trade_id ON trade_management_decisions(trade_id);
CREATE INDEX IF NOT EXISTS idx_tmd_guard_id ON trade_management_decisions(guard_id);
CREATE INDEX IF NOT EXISTS idx_tmd_fired_at ON trade_management_decisions(fired_at);

-- Seed: EURUSD BUY failed-followthrough guard (monitor mode)
INSERT INTO trade_management_guards (
    name,
    guard_type,
    enabled,
    mode,
    applies_to_strategies,
    epic_filter,
    direction_filter,
    condition_config,
    notes
) VALUES (
    'failed_followthrough_eurusd_buy',
    'failed_followthrough',
    TRUE,
    'monitor',
    ARRAY['SMC_SIMPLE'],
    ARRAY['CS.D.EURUSD.CEEM.IP'],
    ARRAY['BUY'],
    '{
        "max_age_minutes": 30,
        "min_mfe_pips": 3.0,
        "adverse_trigger_pips": 6.0
    }',
    'Close EURUSD BUY early when age<=30min, MFE<3 pips, adverse>=6 pips. Enable active mode after 1-week monitor review.'
) ON CONFLICT (name) DO NOTHING;
