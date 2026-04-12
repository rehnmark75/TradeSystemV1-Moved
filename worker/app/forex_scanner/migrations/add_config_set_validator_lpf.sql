-- =============================================================================
-- Migration: Extend config_set split to TradeValidator, LPF, and Claude AI
--
-- This migration adds config_set='live'|'demo' isolation to:
--   1. scanner_global_config   (TradeValidator thresholds + Claude AI settings)
--   2. loss_prevention_config  (LPF master switch, block_mode, penalty_threshold)
--   3. loss_prevention_rules   (individual LPF rules)
--   4. loss_prevention_pair_config (per-pair LPF overrides)
--
-- Each table gains a config_set column. The existing rows are tagged 'live'.
-- A full copy is cloned as 'demo' so both environments start identical and can
-- diverge independently. Promote with: promote_config scanner/lpf promote --confirm
--
-- Idempotent: safe to re-run.
-- =============================================================================

-- =============================================================================
-- 1. scanner_global_config
-- =============================================================================

ALTER TABLE scanner_global_config
    ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live';

-- Tag existing active row as live
UPDATE scanner_global_config
SET config_set = 'live'
WHERE is_active = TRUE AND config_set = 'live';  -- no-op if already set; ensures tag

-- Unique: only one active row per config_set
DROP INDEX IF EXISTS idx_scanner_config_active_per_set;
CREATE UNIQUE INDEX idx_scanner_config_active_per_set
    ON scanner_global_config(config_set)
    WHERE is_active = TRUE;

-- Clone live row as demo (skip if already exists)
DO $$
DECLARE
    source_row scanner_global_config%ROWTYPE;
BEGIN
    -- Check if demo row already exists
    IF EXISTS (SELECT 1 FROM scanner_global_config WHERE config_set = 'demo') THEN
        RAISE NOTICE 'scanner_global_config: demo row already exists, skipping clone';
        RETURN;
    END IF;

    SELECT * INTO source_row
    FROM scanner_global_config
    WHERE is_active = TRUE AND config_set = 'live'
    LIMIT 1;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No active live row found in scanner_global_config';
    END IF;

    -- Override identity/meta fields for the new demo row
    source_row.id         := nextval('scanner_global_config_id_seq');
    source_row.config_set := 'demo';
    source_row.created_at := NOW();
    source_row.updated_at := NOW();
    source_row.updated_by := 'migration_add_config_set_validator_lpf';
    source_row.change_reason := 'Cloned from live by migration add_config_set_validator_lpf';

    INSERT INTO scanner_global_config VALUES (source_row.*);
    RAISE NOTICE 'scanner_global_config: demo row created (id=%)', source_row.id;
END $$;


-- =============================================================================
-- 2. loss_prevention_config
-- =============================================================================

ALTER TABLE loss_prevention_config
    ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live';

-- Tag the existing singleton row as live
UPDATE loss_prevention_config SET config_set = 'live' WHERE id = 1;

-- Unique index per config_set
DROP INDEX IF EXISTS idx_lpf_config_set;
CREATE UNIQUE INDEX idx_lpf_config_set ON loss_prevention_config(config_set);

-- Clone live -> demo
DO $$
DECLARE
    source_row loss_prevention_config%ROWTYPE;
BEGIN
    IF EXISTS (SELECT 1 FROM loss_prevention_config WHERE config_set = 'demo') THEN
        RAISE NOTICE 'loss_prevention_config: demo row already exists, skipping clone';
        RETURN;
    END IF;

    SELECT * INTO source_row FROM loss_prevention_config WHERE config_set = 'live' LIMIT 1;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No live row found in loss_prevention_config';
    END IF;

    source_row.id         := nextval('loss_prevention_config_id_seq');
    source_row.config_set := 'demo';
    source_row.created_at := NOW();
    source_row.updated_at := NOW();

    INSERT INTO loss_prevention_config VALUES (source_row.*);
    RAISE NOTICE 'loss_prevention_config: demo row created (id=%)', source_row.id;
END $$;


-- =============================================================================
-- 3. loss_prevention_rules
-- =============================================================================

ALTER TABLE loss_prevention_rules
    ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live';

-- Tag existing rows as live
UPDATE loss_prevention_rules SET config_set = 'live' WHERE config_set = 'live';  -- no-op

-- The existing UNIQUE constraint on rule_name must become (rule_name, config_set)
-- so that identical rule names can exist in both live and demo sets.
ALTER TABLE loss_prevention_rules
    DROP CONSTRAINT IF EXISTS loss_prevention_rules_rule_name_key;

DROP INDEX IF EXISTS idx_lpf_rules_rule_name_config_set;
CREATE UNIQUE INDEX idx_lpf_rules_rule_name_config_set
    ON loss_prevention_rules(rule_name, config_set);

-- Index for fast per-set queries
DROP INDEX IF EXISTS idx_lpf_rules_config_set;
CREATE INDEX idx_lpf_rules_config_set ON loss_prevention_rules(config_set);

-- Clone all live rules as demo (skip if demo rows already exist)
DO $$
DECLARE
    source_row loss_prevention_rules%ROWTYPE;
BEGIN
    IF EXISTS (SELECT 1 FROM loss_prevention_rules WHERE config_set = 'demo' LIMIT 1) THEN
        RAISE NOTICE 'loss_prevention_rules: demo rows already exist, skipping clone';
        RETURN;
    END IF;

    FOR source_row IN
        SELECT * FROM loss_prevention_rules WHERE config_set = 'live'
    LOOP
        source_row.id         := nextval('loss_prevention_rules_id_seq');
        source_row.config_set := 'demo';
        source_row.created_at := NOW();
        source_row.updated_at := NOW();
        INSERT INTO loss_prevention_rules VALUES (source_row.*);
    END LOOP;

    RAISE NOTICE 'loss_prevention_rules: demo rows created';
END $$;


-- =============================================================================
-- 4. loss_prevention_pair_config
-- =============================================================================

ALTER TABLE loss_prevention_pair_config
    ADD COLUMN IF NOT EXISTS config_set VARCHAR(20) NOT NULL DEFAULT 'live';

-- Tag existing rows as live
UPDATE loss_prevention_pair_config SET config_set = 'live' WHERE config_set = 'live';  -- no-op

-- The existing UNIQUE constraint on epic must become (epic, config_set)
ALTER TABLE loss_prevention_pair_config
    DROP CONSTRAINT IF EXISTS loss_prevention_pair_config_epic_key;

DROP INDEX IF EXISTS idx_lpf_pair_config_epic_config_set;
CREATE UNIQUE INDEX idx_lpf_pair_config_epic_config_set
    ON loss_prevention_pair_config(epic, config_set);

-- Index for fast per-set queries
DROP INDEX IF EXISTS idx_lpf_pair_config_set;
CREATE INDEX idx_lpf_pair_config_set ON loss_prevention_pair_config(config_set);

-- Clone all live pair configs as demo (skip if demo rows already exist)
DO $$
DECLARE
    source_row loss_prevention_pair_config%ROWTYPE;
BEGIN
    IF EXISTS (SELECT 1 FROM loss_prevention_pair_config WHERE config_set = 'demo' LIMIT 1) THEN
        RAISE NOTICE 'loss_prevention_pair_config: demo rows already exist, skipping clone';
        RETURN;
    END IF;

    FOR source_row IN
        SELECT * FROM loss_prevention_pair_config WHERE config_set = 'live'
    LOOP
        source_row.id         := nextval('loss_prevention_pair_config_id_seq');
        source_row.config_set := 'demo';
        source_row.created_at := NOW();
        source_row.updated_at := NOW();
        INSERT INTO loss_prevention_pair_config VALUES (source_row.*);
    END LOOP;

    RAISE NOTICE 'loss_prevention_pair_config: demo rows created';
END $$;


-- =============================================================================
-- Verification
-- =============================================================================

SELECT 'scanner_global_config' AS tbl, config_set, COUNT(*) AS rows
FROM scanner_global_config GROUP BY config_set
UNION ALL
SELECT 'loss_prevention_config', config_set, COUNT(*)
FROM loss_prevention_config GROUP BY config_set
UNION ALL
SELECT 'loss_prevention_rules', config_set, COUNT(*)
FROM loss_prevention_rules GROUP BY config_set
UNION ALL
SELECT 'loss_prevention_pair_config', config_set, COUNT(*)
FROM loss_prevention_pair_config GROUP BY config_set
ORDER BY tbl, config_set;
